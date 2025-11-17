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
from transformers import Wav2Vec2Config, Wav2Vec2Model
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


class LearnablePositionalEncoding(nn.Module):
    """학습 가능한 포지셔널 인코딩"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, 1, d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positional = self.pe[:seq_len].transpose(0, 1)  # (1, seq_len, d_model)
        return x + positional


class VoiceEmotionWav2Vec2(nn.Module):
    """Wav2Vec2 + Transformer 기반 음성 감정 분류 모델"""
    
    def __init__(
        self,
        num_emotions: int = 5,
        transformer_config: Optional[Dict[str, Any]] = None
    ):
        super(VoiceEmotionWav2Vec2, self).__init__()
        
        transformer_config = transformer_config or {}
        self.wav2vec2 = Wav2Vec2Model(Wav2Vec2Config())
        
        d_model = transformer_config.get("d_model", 512)
        nhead = transformer_config.get("nhead", 8)
        num_layers = transformer_config.get("num_layers", 4)
        dropout_rate = transformer_config.get("dropout", 0.1)
        
        # Wav2Vec2 출력(768)을 Transformer 입력 차원으로 투영
        self.input_projection = nn.Linear(
            self.wav2vec2.config.hidden_size,
            d_model
        )
        self.pos_encoding = LearnablePositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        hidden_dim = max(d_model // 2, num_emotions)
        bottleneck_dim = max(d_model // 4, num_emotions)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(bottleneck_dim, num_emotions)
        )
    
    def forward(self, audio_inputs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """음성 입력 텐서로부터 감정 로짓 계산"""
        outputs = self.wav2vec2(
            audio_inputs,
            attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)
        
        projected = self.input_projection(hidden_states)
        projected = self.pos_encoding(projected)
        encoded = self.transformer(projected)
        pooled = encoded.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

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
        self.model_variant = None  # wav2vec2 또는 cnn
        self.feature_extractor = AudioFeatureExtractor(sample_rate)
        self.emotions = ['happy', 'depressed', 'surprised', 'angry', 'neutral']
        
        self._load_model()
        
    def _load_model(self):
        """음성 감정 분석 모델 로드"""
        try:
            checkpoint = None
            state_dict = None
            is_wav2vec2_checkpoint = False
            
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    # 전체 모델이 저장된 경우 (직렬화된 nn.Module)
                    self.model = checkpoint
                    if hasattr(self.model, 'wav2vec2'):
                        self.model_variant = "wav2vec2"
                    else:
                        self.model_variant = "cnn"
            else:
                logger.warning(f"음성 감정 모델 파일을 찾을 수 없습니다: {self.model_path}")
            
            if self.model is None and state_dict is not None:
                is_wav2vec2_checkpoint = any(
                    key.startswith('wav2vec2.') for key in state_dict.keys()
                )
                if is_wav2vec2_checkpoint:
                    transformer_config = {}
                    if isinstance(checkpoint, dict):
                        transformer_config = checkpoint.get('model_config', {}) or {}
                    try:
                        self.model = VoiceEmotionWav2Vec2(
                            num_emotions=len(self.emotions),
                            transformer_config=transformer_config
                        )
                        self.model.load_state_dict(state_dict, strict=True)
                        self.model_variant = "wav2vec2"
                        logger.info("Wav2Vec2 기반 음성 감정 모델을 로드했습니다.")
                    except Exception as wav_err:
                        logger.error(f"Wav2Vec2 모델 로드 실패: {wav_err}")
                        self.model = None
                
            if self.model is None:
                # CNN 백업 모델
                self.model = VoiceEmotionCNN()
                if state_dict is not None and not is_wav2vec2_checkpoint:
                    try:
                        self.model.load_state_dict(state_dict)
                    except Exception as cnn_err:
                        logger.warning(f"CNN 가중치 로드 실패, 기본 가중치 사용: {cnn_err}")
                self.model_variant = "cnn"
                logger.info("CNN 백업 음성 모델을 사용합니다.")
            
            if self.model is not None:
                self.model.to(self.device)
                self.model.eval()
                
        except Exception as e:
            logger.error(f"음성 감정 모델 로드 실패: {e}")
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
            if self.model_variant == "wav2vec2":
                return self._analyze_with_wav2vec2(processed_audio)
            
            # 특징 추출 (CNN 백업용)
            features = self._extract_features_for_model(processed_audio)
            if features is None:
                return None
            
            input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
            return probs
            
        except Exception as e:
            logger.error(f"음성 감정 분석 실패: {e}")
            return None
    
    def _analyze_with_wav2vec2(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """Wav2Vec2 기반 모델 추론"""
        try:
            audio_tensor = torch.tensor(
                audio_data,
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)
            attention_mask = torch.ones(
                audio_tensor.shape,
                dtype=torch.long,
                device=self.device
            )
            
            with torch.no_grad():
                outputs = self.model(audio_tensor, attention_mask=attention_mask)
                probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
            return probs
        except Exception as e:
            logger.error(f"Wav2Vec2 음성 감정 분석 실패: {e}")
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
