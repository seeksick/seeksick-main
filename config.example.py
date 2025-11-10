#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
설정 파일 예제
실제 사용 시 config.py로 복사하여 사용하세요.
"""

# OpenAI API 키
OPENAI_API_KEY = "your-api-key-here"

# Flask 설정
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False

# 감정 분석 설정
FUSION_INTERVAL = 5.0  # Late Fusion 간격 (초)

# 모델 경로 (기본값)
MODEL_PATHS = {
    "face": "models/seeksick-resnet18.pth",
    "voice": "models/seeksick-voice.pt",
    "text": "models/seeksick-kobert.pt"
}

