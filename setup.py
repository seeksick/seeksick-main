#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Seeksick 멀티모달 감정 분석 프로그램 설치 스크립트
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Python 버전 확인"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 이상이 필요합니다.")
        print(f"현재 버전: {sys.version}")
        sys.exit(1)
    else:
        print(f"✅ Python 버전 확인 완료: {sys.version.split()[0]}")

def install_requirements():
    """필요한 패키지 설치"""
    print("\n📦 필요한 패키지를 설치합니다...")
    
    try:
        # pip 업그레이드
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # requirements.txt 설치
        requirements_file = Path(__file__).parent / "requirements.txt"
        if requirements_file.exists():
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            print("✅ 패키지 설치 완료")
        else:
            print("❌ requirements.txt 파일을 찾을 수 없습니다.")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 패키지 설치 실패: {e}")
        sys.exit(1)

def create_model_directories():
    """모델 디렉토리 생성"""
    print("\n📁 모델 디렉토리를 생성합니다...")
    
    base_dir = Path(__file__).parent
    model_dir = base_dir / "models"
    
    model_dir.mkdir(exist_ok=True)
    
    # 모델 파일 체크
    model_files = [
        "seeksick-resnet18.pth",
        "seeksick-voice.pt", 
        "seeksick-kobert.pt"
    ]
    
    missing_models = []
    for model_file in model_files:
        model_path = model_dir / model_file
        if model_path.exists():
            print(f"✅ {model_file} 발견")
        else:
            print(f"⚠️ {model_file} 없음 (나중에 추가 필요)")
            missing_models.append(model_file)
    
    if missing_models:
        print(f"\n📌 누락된 모델 파일: {', '.join(missing_models)}")
        print("📌 모델 파일들을 models/ 디렉토리에 배치해주세요.")
    
    print("✅ 모델 디렉토리 설정 완료")

def check_system_dependencies():
    """시스템 의존성 확인"""
    print("\n🔍 시스템 의존성을 확인합니다...")
    
    # OpenCV 카메라 접근 가능 여부
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ 웹캠 접근 가능")
            cap.release()
        else:
            print("⚠️ 웹캠에 접근할 수 없습니다. 카메라 권한을 확인해주세요.")
    except ImportError:
        print("⚠️ OpenCV를 먼저 설치해주세요.")
    
    # 마이크 접근 가능 여부
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if input_devices:
            print("✅ 마이크 접근 가능")
        else:
            print("⚠️ 사용 가능한 마이크를 찾을 수 없습니다.")
    except ImportError:
        print("⚠️ sounddevice를 먼저 설치해주세요.")

def create_example_config():
    """예제 설정 파일 생성"""
    print("\n⚙️ 예제 설정 파일을 생성합니다...")
    
    config_content = """# Seeksick 멀티모달 감정 분석 설정

# 모델 파일 경로
FACE_MODEL_PATH=models/seeksick-resnet18.pth
VOICE_MODEL_PATH=models/seeksick-voice.pt
TEXT_MODEL_PATH=models/seeksick-kobert.pt

# 오디오 설정
SAMPLE_RATE=16000
AUDIO_BUFFER_SIZE=3.0

# 비디오 설정
VIDEO_FPS=10
FACE_DETECTION_SCALE=1.1
FACE_MIN_NEIGHBORS=5

# 처리 설정
CONFIDENCE_THRESHOLD=0.5
MAX_TEXT_LENGTH=128

# 로깅 설정
LOG_LEVEL=INFO
LOG_FILE=logs/emotion_analysis.log
"""
    
    config_path = Path(__file__).parent / ".env"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print("✅ .env 설정 파일 생성 완료")

def main():
    """메인 설치 함수"""
    print("🚀 Seeksick 멀티모달 감정 분석 프로그램 설치를 시작합니다.")
    print("=" * 60)
    
    # 1. Python 버전 확인
    check_python_version()
    
    # 2. 패키지 설치
    install_requirements()
    
    # 3. 모델 디렉토리 생성
    create_model_directories()
    
    # 4. 시스템 의존성 확인
    check_system_dependencies()
    
    # 5. 설정 파일 생성
    create_example_config()
    
    print("\n" + "=" * 60)
    print("🎉 설치가 완료되었습니다!")
    print("\n📋 다음 단계:")
    print("1. 모델 파일들을 models/ 디렉토리에 배치하세요:")
    print("   - seeksick-resnet18.pth (얼굴 감정 모델)")
    print("   - seeksick-voice.pt (음성 감정 모델)")
    print("   - seeksick-kobert.pt (텍스트 감정 모델)")
    print("\n2. 프로그램을 실행하세요:")
    print("   python main.py")
    print("\n3. 개별 모델 테스트:")
    print("   python -m models.face_emotion_model")
    print("   python -m models.voice_emotion_model")
    print("   python -m models.text_emotion_model")

if __name__ == "__main__":
    main()
