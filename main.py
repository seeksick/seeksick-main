#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
멀티모달 감정 분석 메인 프로그램

이 프로그램은 다음 3가지 모달리티에서 감정을 분석합니다:
1. 얼굴 감정 (seeksick-resnet18.pth)
2. 음성 감정 (seeksick-voice.pt)
3. 텍스트 감정 (seeksick-kobert.pt)

모든 감정은 [행복, 우울, 놀람, 화남, 중립] 5가지로 분류됩니다.
"""

import os
import sys
import time
import threading
import queue
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import sounddevice as sd
from scipy.io import wavfile
import whisper

# 로컬 모델 임포트
from models import FaceEmotionAnalyzer, VoiceEmotionAnalyzer, TextEmotionAnalyzer, EMOTIONS

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 전역 상수
SAMPLE_RATE = 16000  # Whisper 권장 샘플레이트
AUDIO_BUFFER_SIZE = 3.0  # 3초 단위로 음성 분석
VIDEO_FPS = 10  # 얼굴 감정 분석 주기

class AudioRecorder:
    """음성 녹음 및 처리 클래스"""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, buffer_duration: float = AUDIO_BUFFER_SIZE):
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.buffer_size = int(sample_rate * buffer_duration)
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio_data = []
        
    def audio_callback(self, indata, frames, time, status):
        """음성 데이터 콜백 함수"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        self.audio_data.extend(indata[:, 0])  # 모노 채널만 사용
        
        # 버퍼가 가득 찼을 때 큐에 추가
        if len(self.audio_data) >= self.buffer_size:
            audio_chunk = np.array(self.audio_data[:self.buffer_size], dtype=np.float32)
            self.audio_queue.put(audio_chunk)
            self.audio_data = self.audio_data[self.buffer_size:]
    
    def start_recording(self):
        """음성 녹음 시작"""
        self.is_recording = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=1024
        )
        self.stream.start()
        logger.info("음성 녹음이 시작되었습니다.")
    
    def stop_recording(self):
        """음성 녹음 중지"""
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        logger.info("음성 녹음이 중지되었습니다.")
    
    def get_audio_chunk(self) -> Optional[np.ndarray]:
        """음성 청크 가져오기"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

# 분석기 클래스들은 models 패키지에서 임포트하여 사용

class MultimodalEmotionAnalyzer:
    """멀티모달 감정 분석 메인 클래스"""
    
    def __init__(self):
        self.audio_recorder = AudioRecorder()
        self.face_analyzer = FaceEmotionAnalyzer()
        self.voice_analyzer = VoiceEmotionAnalyzer()
        self.text_analyzer = TextEmotionAnalyzer()
        
        # Whisper 모델 로드
        self.whisper_model = whisper.load_model("base")
        logger.info("Whisper 모델을 로드했습니다.")
        
        # 비디오 캡처
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error("웹캠을 열 수 없습니다.")
            sys.exit(1)
            
        # 결과 저장
        self.results = {
            'face': [],
            'voice': [],
            'text': [],
            'timestamps': []
        }
        
        self.is_running = False
        
    def transcribe_audio(self, audio_data: np.ndarray) -> str:
        """음성을 텍스트로 변환"""
        try:
            # Whisper는 float32 배열을 입력으로 받음
            result = self.whisper_model.transcribe(audio_data)
            text = result["text"].strip()
            logger.info(f"음성 인식 결과: {text}")
            return text
        except Exception as e:
            logger.error(f"음성 인식 실패: {e}")
            return ""
    
    def process_audio_thread(self):
        """음성 처리 스레드"""
        while self.is_running:
            audio_chunk = self.audio_recorder.get_audio_chunk()
            if audio_chunk is not None:
                timestamp = datetime.now()
                
                # 1. 음성에서 텍스트 추출
                text = self.transcribe_audio(audio_chunk)
                
                # 2. 음성 감정 분석
                voice_probs = self.voice_analyzer.analyze_voice(audio_chunk)
                
                # 3. 텍스트 감정 분석
                text_probs = self.text_analyzer.analyze_text(text) if text else None
                
                # 결과 저장
                self.results['voice'].append(voice_probs)
                self.results['text'].append(text_probs)
                self.results['timestamps'].append(timestamp)
                
                # 결과 출력
                self.print_emotion_results(timestamp, voice_probs, text_probs, text)
            
            time.sleep(0.1)
    
    def print_emotion_results(self, timestamp, voice_probs, text_probs, text=""):
        """감정 분석 결과 출력"""
        print(f"\n{'='*60}")
        print(f"시간: {timestamp.strftime('%H:%M:%S')}")
        print(f"텍스트: {text}")
        print(f"{'='*60}")
        
        if voice_probs is not None:
            print("🎤 음성 감정 분석:")
            for i, (emotion, prob) in enumerate(zip(EMOTIONS, voice_probs)):
                print(f"  {emotion}: {prob:.3f}")
        
        if text_probs is not None:
            print("📝 텍스트 감정 분석:")
            for i, (emotion, prob) in enumerate(zip(EMOTIONS, text_probs)):
                print(f"  {emotion}: {prob:.3f}")
    
    def run(self):
        """메인 실행 함수"""
        print("멀티모달 감정 분석을 시작합니다...")
        print("종료하려면 'q'를 누르세요.")
        
        self.is_running = True
        
        # 음성 녹음 시작
        self.audio_recorder.start_recording()
        
        # 음성 처리 스레드 시작
        audio_thread = threading.Thread(target=self.process_audio_thread)
        audio_thread.daemon = True
        audio_thread.start()
        
        try:
            frame_count = 0
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # 얼굴 감정 분석 (주기적으로)
                if frame_count % (30 // VIDEO_FPS) == 0:  # 30fps 기준
                    face_probs = self.face_analyzer.analyze_face(frame)
                    if face_probs is not None:
                        timestamp = datetime.now()
                        self.results['face'].append(face_probs)
                        
                        print(f"\n👤 얼굴 감정 분석 ({timestamp.strftime('%H:%M:%S')}):")
                        for emotion, prob in zip(EMOTIONS, face_probs):
                            print(f"  {emotion}: {prob:.3f}")
                
                # 웹캠 화면에 감정 정보 표시
                self.display_frame_with_emotions(frame)
                
                # 화면 출력
                cv2.imshow('Multimodal Emotion Analysis', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                frame_count += 1
                
        finally:
            self.cleanup()
    
    def display_frame_with_emotions(self, frame):
        """프레임에 감정 정보 표시"""
        # 최근 감정 분석 결과가 있으면 화면에 표시
        if self.results['face']:
            face_probs = self.results['face'][-1]
            max_idx = np.argmax(face_probs)
            emotion = EMOTIONS[max_idx]
            confidence = face_probs[max_idx]
            
            # 텍스트 표시
            text = f"Face: {emotion} ({confidence:.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.results['voice']:
            voice_probs = self.results['voice'][-1]
            if voice_probs is not None:
                max_idx = np.argmax(voice_probs)
                emotion = EMOTIONS[max_idx]
                confidence = voice_probs[max_idx]
                
                text = f"Voice: {emotion} ({confidence:.2f})"
                cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if self.results['text']:
            text_probs = self.results['text'][-1]
            if text_probs is not None:
                max_idx = np.argmax(text_probs)
                emotion = EMOTIONS[max_idx]
                confidence = text_probs[max_idx]
                
                text = f"Text: {emotion} ({confidence:.2f})"
                cv2.putText(frame, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    def cleanup(self):
        """리소스 정리"""
        self.is_running = False
        self.audio_recorder.stop_recording()
        self.cap.release()
        cv2.destroyAllWindows()
        
        print("\n프로그램이 종료되었습니다.")
        
        # 결과 요약 출력
        self.print_summary()
    
    def print_summary(self):
        """분석 결과 요약 출력"""
        print(f"\n{'='*60}")
        print("분석 결과 요약")
        print(f"{'='*60}")
        print(f"총 얼굴 감정 분석 횟수: {len(self.results['face'])}")
        print(f"총 음성 감정 분석 횟수: {len([x for x in self.results['voice'] if x is not None])}")
        print(f"총 텍스트 감정 분석 횟수: {len([x for x in self.results['text'] if x is not None])}")

def main():
    """메인 함수"""
    try:
        analyzer = MultimodalEmotionAnalyzer()
        analyzer.run()
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
