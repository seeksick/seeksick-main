#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
얼굴 감정 분석 모델 (seeksick-resnet18.pth)

ResNet18 기반의 얼굴 감정 분류 모델
감정: [행복, 우울, 놀람, 화남, 중립]
"""

import os
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class FaceEmotionResNet18(nn.Module):
    """ResNet18 기반 얼굴 감정 분류 모델"""
    
    def __init__(self, num_emotions: int = 5, pretrained: bool = False):
        super(FaceEmotionResNet18, self).__init__()
        
        # ResNet18 백본
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # 분류기 교체 (5개 감정)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_emotions)
        
        # 드롭아웃 추가
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 백본 통과 (fc 레이어 제외)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 드롭아웃 적용
        x = self.dropout(x)
        
        # 최종 분류
        x = self.backbone.fc(x)
        
        return x

class FaceEmotionAnalyzer:
    """얼굴 감정 분석기"""
    
    def __init__(self, model_path: str = "models/seeksick-resnet18.pth"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.face_cascade = None
        self.preprocess = None
        self.emotions = ['happy', 'depressed', 'surprised', 'angry', 'neutral']
        
        self._load_model()
        self._load_face_detector()
        self._setup_preprocessing()
        
    def _load_model(self):
        """얼굴 감정 분석 모델 로드"""
        try:
            self.model = FaceEmotionResNet18(num_emotions=5)
            
            if os.path.exists(self.model_path):
                # 모델 가중치 로드
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # state_dict 형태에 따라 처리
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                    
                logger.info(f"얼굴 감정 모델을 로드했습니다: {self.model_path}")
            else:
                logger.warning(f"얼굴 감정 모델 파일을 찾을 수 없습니다: {self.model_path}")
                logger.info("임의의 가중치로 모델을 초기화합니다.")
                
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"얼굴 감정 모델 로드 실패: {e}")
            self.model = None
    
    def _load_face_detector(self):
        """얼굴 검출기 로드"""
        try:
            # Haar Cascade 분류기 로드
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                logger.error("얼굴 검출기 로드 실패")
                self.face_cascade = None
            else:
                logger.info("얼굴 검출기를 로드했습니다.")
                
        except Exception as e:
            logger.error(f"얼굴 검출기 로드 실패: {e}")
            self.face_cascade = None
    
    def _setup_preprocessing(self):
        """전처리 파이프라인 설정"""
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet 평균
                std=[0.229, 0.224, 0.225]   # ImageNet 표준편차
            )
        ])
    
    def detect_faces(self, frame: np.ndarray) -> list:
        """프레임에서 얼굴 검출"""
        if self.face_cascade is None:
            return []
            
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 검출
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def extract_face_roi(self, frame: np.ndarray, face_coords: tuple) -> np.ndarray:
        """얼굴 영역 추출"""
        x, y, w, h = face_coords
        
        # 여백 추가 (10%)
        margin = 0.1
        mx = int(w * margin)
        my = int(h * margin)
        
        # 경계 확인
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(frame.shape[1], x + w + mx)
        y2 = min(frame.shape[0], y + h + my)
        
        face_roi = frame[y1:y2, x1:x2]
        return face_roi
    
    def analyze_face(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, tuple]]:
        """얼굴 감정 분석"""
        if self.model is None:
            return None
            
        try:
            # 얼굴 검출
            faces = self.detect_faces(frame)
            
            if len(faces) == 0:
                return None
                
            # 가장 큰 얼굴 선택
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            
            # 얼굴 영역 추출
            face_roi = self.extract_face_roi(frame, largest_face)
            
            if face_roi.size == 0:
                return None
            
            # 전처리
            input_tensor = self.preprocess(face_roi).unsqueeze(0).to(self.device)
            
            # 추론
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
                
            return probs, largest_face
            
        except Exception as e:
            logger.error(f"얼굴 감정 분석 실패: {e}")
            return None
    
    def get_emotion_label(self, probs: np.ndarray) -> Tuple[str, float]:
        """확률에서 감정 라벨 추출"""
        max_idx = np.argmax(probs)
        emotion = self.emotions[max_idx]
        confidence = probs[max_idx]
        return emotion, confidence
    
    def draw_emotion_on_frame(self, frame: np.ndarray, face_coords: tuple, 
                            probs: np.ndarray) -> np.ndarray:
        """프레임에 감정 정보 그리기"""
        x, y, w, h = face_coords
        emotion, confidence = self.get_emotion_label(probs)
        
        # 얼굴 박스 그리기
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 감정 텍스트 그리기
        label = f"{emotion}: {confidence:.2f}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        # 감정별 확률 막대 그래프
        bar_width = w
        bar_height = 10
        start_y = y + h + 20
        
        for i, (emo, prob) in enumerate(zip(self.emotions, probs)):
            bar_x = x
            bar_y = start_y + i * (bar_height + 5)
            bar_w = int(bar_width * prob)
            
            # 배경 (회색)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (128, 128, 128), -1)
            
            # 확률 막대 (색상)
            colors = [(0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)]
            color = colors[i] if i < len(colors) else (255, 255, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_height), 
                         color, -1)
            
            # 라벨
            cv2.putText(frame, f"{emo}: {prob:.2f}", (bar_x + bar_width + 10, bar_y + 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame

def test_face_emotion_analyzer():
    """얼굴 감정 분석기 테스트"""
    analyzer = FaceEmotionAnalyzer()
    
    # 웹캠 테스트
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    print("얼굴 감정 분석 테스트를 시작합니다. 'q'를 눌러 종료하세요.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = analyzer.analyze_face(frame)
            
            if result is not None:
                probs, face_coords = result
                frame = analyzer.draw_emotion_on_frame(frame, face_coords, probs)
                
                emotion, confidence = analyzer.get_emotion_label(probs)
                print(f"감정: {emotion} (신뢰도: {confidence:.3f})")
            
            cv2.imshow('Face Emotion Analysis Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_face_emotion_analyzer()
