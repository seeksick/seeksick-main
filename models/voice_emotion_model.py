#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
음성 감정 분석 모델 (seeksick-voice.pt)

Wav2Vec2 또는 CNN 기반의 음성 감정 분류 모델
감정: [행복, 우울, 놀람, 화남, 중립]
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from typing import Optional, Dict, Any
import sounddevice as sd
from scipy.io import wavfile

logger = logging.getLogger(__name__)

class VoiceEmotionCNN(nn.Module):
    """CNN 기반 음성 감정 분류 모델"""
    
    def __init__(self, input_dim: int = 128, num_emotions: int = 5):
        super(VoiceEmotionCNN, self).__init__()
        
        # 1D CNN 레이어들
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
        
        # 풀링 레이어
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 드롭아웃
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully Connected 레이어들
        self.fc1 = nn.Linear(256 * (input_dim // 8), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_emotions)
        
        # 배치 정규화
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length) -> (batch_size, 1, sequence_length)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x

class AudioFeatureExtractor:
    """오디오 특징 추출기"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def extract_mfcc(self, audio_data: np.ndarray, n_mfcc: int = 13, 
                     n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
        """MFCC 특징 추출"""
        try:
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length
            )
            return mfccs
        except Exception as e:
            logger.error(f"MFCC 추출 실패: {e}")
            return np.zeros((n_mfcc, 1))
    
    def extract_mel_spectrogram(self, audio_data: np.ndarray, 
                               n_mels: int = 128) -> np.ndarray:
        """Mel Spectrogram 특징 추출"""
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=self.sample_rate,
                n_mels=n_mels,
                fmax=self.sample_rate // 2
            )
            # 로그 스케일 변환
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            return mel_spec_db
        except Exception as e:
            logger.error(f"Mel Spectrogram 추출 실패: {e}")
            return np.zeros((n_mels, 1))
    
    def extract_spectral_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """스펙트럴 특징 추출"""
        try:
            # 스펙트럴 중심
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data, sr=self.sample_rate
            )[0]
            
            # 스펙트럴 롤오프
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=self.sample_rate
            )[0]
            
            # 영교차율
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
            
            # 크로마 특징
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate)
            
            features = {
                'spectral_centroid_mean': np.mean(spectral_centroids),
                'spectral_centroid_std': np.std(spectral_centroids),
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'spectral_rolloff_std': np.std(spectral_rolloff),
                'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
                'zero_crossing_rate_std': np.std(zero_crossing_rate),
                'chroma_mean': np.mean(chroma),
                'chroma_std': np.std(chroma)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"스펙트럴 특징 추출 실패: {e}")
            return {}
    
    def preprocess_audio(self, audio_data: np.ndarray, 
                        target_length: int = None) -> np.ndarray:
        """오디오 데이터 전처리"""
        # 정규화
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # 길이 조정
        if target_length is not None:
            if len(audio_data) > target_length:
                # 자르기
                audio_data = audio_data[:target_length]
            elif len(audio_data) < target_length:
                # 패딩
                pad_length = target_length - len(audio_data)
                audio_data = np.pad(audio_data, (0, pad_length), mode='constant')
        
        return audio_data

class VoiceEmotionAnalyzer:
    """음성 감정 분석기"""
    
    def __init__(self, model_path: str = "models/seeksick-voice.pt", 
                 sample_rate: int = 16000):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.feature_extractor = AudioFeatureExtractor(sample_rate)
        self.emotions = ['happy', 'depressed', 'surprised', 'angry', 'neutral']
        
        self._load_model()
        
    def _load_model(self):
        """음성 감정 분석 모델 로드"""
        try:
            if os.path.exists(self.model_path):
                # 모델 파일이 존재하는 경우
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                if isinstance(checkpoint, dict):
                    # 체크포인트 형태
                    if 'model_state_dict' in checkpoint:
                        self.model = VoiceEmotionCNN()
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'model' in checkpoint:
                        self.model = checkpoint['model']
                    else:
                        # state_dict만 있는 경우
                        self.model = VoiceEmotionCNN()
                        self.model.load_state_dict(checkpoint)
                else:
                    # 전체 모델이 저장된 경우
                    self.model = checkpoint
                
                logger.info(f"음성 감정 모델을 로드했습니다: {self.model_path}")
            else:
                # 모델 파일이 없는 경우 기본 모델 사용
                logger.warning(f"음성 감정 모델 파일을 찾을 수 없습니다: {self.model_path}")
                logger.info("기본 CNN 모델을 초기화합니다.")
                self.model = VoiceEmotionCNN()
                
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"음성 감정 모델 로드 실패: {e}")
            # 백업으로 기본 모델 사용
            try:
                self.model = VoiceEmotionCNN()
                self.model.to(self.device)
                self.model.eval()
                logger.info("백업 모델을 사용합니다.")
            except Exception as backup_e:
                logger.error(f"백업 모델 로드도 실패: {backup_e}")
                self.model = None
    
    def analyze_voice(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """음성 감정 분석"""
        if self.model is None:
            return None
            
        try:
            # 오디오 전처리
            processed_audio = self.feature_extractor.preprocess_audio(
                audio_data, target_length=self.sample_rate * 3  # 3초로 고정
            )
            
            # 특징 추출 (여러 방법 시도)
            features = self._extract_features_for_model(processed_audio)
            
            if features is None:
                return None
            
            # 텐서 변환
            input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # 추론
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
                
            return probs
            
        except Exception as e:
            logger.error(f"음성 감정 분석 실패: {e}")
            return None
    
    def _extract_features_for_model(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """모델용 특징 추출"""
        try:
            # 방법 1: Mel Spectrogram의 평균 사용
            mel_spec = self.feature_extractor.extract_mel_spectrogram(audio_data)
            mel_features = np.mean(mel_spec, axis=1)  # 시간축으로 평균
            
            # 길이를 128로 맞춤 (모델 입력 크기)
            if len(mel_features) > 128:
                mel_features = mel_features[:128]
            elif len(mel_features) < 128:
                mel_features = np.pad(mel_features, (0, 128 - len(mel_features)), 
                                    mode='constant')
            
            return mel_features
            
        except Exception as e:
            logger.error(f"특징 추출 실패: {e}")
            
            # 백업 방법: MFCC 사용
            try:
                mfccs = self.feature_extractor.extract_mfcc(audio_data, n_mfcc=13)
                mfcc_features = np.mean(mfccs, axis=1)
                
                # 128 차원으로 확장 (패딩)
                if len(mfcc_features) < 128:
                    mfcc_features = np.pad(mfcc_features, (0, 128 - len(mfcc_features)), 
                                         mode='constant')
                else:
                    mfcc_features = mfcc_features[:128]
                    
                return mfcc_features
                
            except Exception as backup_e:
                logger.error(f"백업 특징 추출도 실패: {backup_e}")
                return None
    
    def get_emotion_label(self, probs: np.ndarray) -> tuple:
        """확률에서 감정 라벨 추출"""
        max_idx = np.argmax(probs)
        emotion = self.emotions[max_idx]
        confidence = probs[max_idx]
        return emotion, confidence
    
    def record_and_analyze(self, duration: float = 3.0) -> Optional[np.ndarray]:
        """실시간 녹음 및 분석"""
        try:
            logger.info(f"{duration}초 동안 음성을 녹음합니다...")
            
            # 녹음
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32
            )
            sd.wait()  # 녹음 완료까지 대기
            
            # 분석
            audio_data = audio_data.flatten()
            return self.analyze_voice(audio_data)
            
        except Exception as e:
            logger.error(f"실시간 녹음 및 분석 실패: {e}")
            return None
    
    def save_audio_sample(self, audio_data: np.ndarray, filename: str):
        """오디오 샘플 저장"""
        try:
            # int16 형태로 변환
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wavfile.write(filename, self.sample_rate, audio_int16)
            logger.info(f"오디오 샘플을 저장했습니다: {filename}")
        except Exception as e:
            logger.error(f"오디오 저장 실패: {e}")

def test_voice_emotion_analyzer():
    """음성 감정 분석기 테스트"""
    analyzer = VoiceEmotionAnalyzer()
    
    print("음성 감정 분석 테스트를 시작합니다.")
    print("아무 키나 누르면 3초간 녹음이 시작됩니다. 'q'를 입력하면 종료합니다.")
    
    while True:
        user_input = input("\n녹음 시작하려면 엔터를 누르세요 (종료: q): ")
        
        if user_input.lower() == 'q':
            break
            
        # 녹음 및 분석
        probs = analyzer.record_and_analyze(duration=3.0)
        
        if probs is not None:
            emotion, confidence = analyzer.get_emotion_label(probs)
            
            print(f"\n=== 음성 감정 분석 결과 ===")
            print(f"예측 감정: {emotion} (신뢰도: {confidence:.3f})")
            print("감정별 확률:")
            for emo, prob in zip(analyzer.emotions, probs):
                print(f"  {emo}: {prob:.3f}")
        else:
            print("음성 감정 분석에 실패했습니다.")
    
    print("음성 감정 분석 테스트를 종료합니다.")

if __name__ == "__main__":
    test_voice_emotion_analyzer()
