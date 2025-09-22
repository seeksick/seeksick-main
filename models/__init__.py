#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Seeksick 멀티모달 감정 분석 모델 패키지

이 패키지는 다음 3가지 모달리티의 감정 분석 모델을 포함합니다:
1. 얼굴 감정 분석 (seeksick-resnet18.pth)
2. 음성 감정 분석 (seeksick-voice.pt)  
3. 텍스트 감정 분석 (seeksick-kobert.pt)

모든 모델은 [행복, 우울, 놀람, 화남, 중립] 5가지 감정을 분류합니다.
"""

from .face_emotion_model import FaceEmotionAnalyzer, FaceEmotionResNet18
from .voice_emotion_model import VoiceEmotionAnalyzer, VoiceEmotionCNN, AudioFeatureExtractor
from .text_emotion_model import TextEmotionAnalyzer, KoBERTForEmotion, TextPreprocessor

__all__ = [
    # 얼굴 감정 분석
    'FaceEmotionAnalyzer',
    'FaceEmotionResNet18',
    
    # 음성 감정 분석
    'VoiceEmotionAnalyzer',
    'VoiceEmotionCNN',
    'AudioFeatureExtractor',
    
    # 텍스트 감정 분석
    'TextEmotionAnalyzer',
    'KoBERTForEmotion',
    'TextPreprocessor',
]

# 버전 정보
__version__ = "1.0.0"

# 감정 라벨
EMOTIONS = ['happy', 'depressed', 'surprised', 'angry', 'neutral']

# 모델 파일 경로
MODEL_PATHS = {
    'face': 'models/seeksick-resnet18.pth',
    'voice': 'models/seeksick-voice.pt',
    'text': 'models/seeksick-kobert.pt'
}
