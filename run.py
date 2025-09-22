#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Seeksick 멀티모달 감정 분석 프로그램 실행 스크립트
"""

import sys
import os
from pathlib import Path

def check_models():
    """모델 파일 존재 여부 확인"""
    model_dir = Path(__file__).parent / "models"
    required_models = [
        "seeksick-resnet18.pth",
        "seeksick-voice.pt", 
        "seeksick-kobert.pt"
    ]
    
    missing_models = []
    for model_file in required_models:
        model_path = model_dir / model_file
        if not model_path.exists():
            missing_models.append(model_file)
    
    if missing_models:
        print("⚠️ 다음 모델 파일들이 누락되었습니다:")
        for model in missing_models:
            print(f"   - {model}")
        print(f"\n📁 모델 파일들을 {model_dir} 디렉토리에 배치해주세요.")
        
        choice = input("\n모델 파일 없이 계속 진행하시겠습니까? (y/N): ")
        if choice.lower() != 'y':
            print("프로그램을 종료합니다.")
            sys.exit(1)
    else:
        print("✅ 모든 모델 파일이 준비되었습니다.")

def check_dependencies():
    """필수 라이브러리 확인"""
    required_packages = [
        'torch', 'torchvision', 'transformers', 
        'sounddevice', 'scipy', 'librosa', 'whisper',
        'opencv-python', 'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 다음 패키지들이 설치되지 않았습니다:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n설치 명령어:")
        print("pip install " + " ".join(missing_packages))
        print("\n또는 다음 명령어로 모든 의존성을 설치하세요:")
        print("python setup.py")
        sys.exit(1)
    else:
        print("✅ 모든 필수 패키지가 설치되었습니다.")

def run_main():
    """메인 프로그램 실행"""
    try:
        from main import main
        print("\n🚀 멀티모달 감정 분석을 시작합니다...")
        print("=" * 60)
        main()
    except KeyboardInterrupt:
        print("\n\n👋 사용자에 의해 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 프로그램 실행 중 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()

def show_menu():
    """메뉴 표시"""
    print("🎯 Seeksick 멀티모달 감정 분석 프로그램")
    print("=" * 50)
    print("1. 멀티모달 감정 분석 실행")
    print("2. 얼굴 감정 분석만 테스트")
    print("3. 음성 감정 분석만 테스트")
    print("4. 텍스트 감정 분석만 테스트")
    print("5. 시스템 상태 확인")
    print("0. 종료")
    print("=" * 50)

def run_individual_test(model_type):
    """개별 모델 테스트 실행"""
    try:
        if model_type == "face":
            from models.face_emotion_model import test_face_emotion_analyzer
            test_face_emotion_analyzer()
        elif model_type == "voice":
            from models.voice_emotion_model import test_voice_emotion_analyzer
            test_voice_emotion_analyzer()
        elif model_type == "text":
            from models.text_emotion_model import test_text_emotion_analyzer
            test_text_emotion_analyzer()
    except Exception as e:
        print(f"❌ {model_type} 모델 테스트 실행 실패: {e}")

def check_system_status():
    """시스템 상태 확인"""
    print("\n🔍 시스템 상태를 확인합니다...")
    print("=" * 40)
    
    # Python 버전
    print(f"Python 버전: {sys.version.split()[0]}")
    
    # GPU 사용 가능 여부
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: 사용 가능 ({torch.cuda.get_device_name(0)})")
        else:
            print("GPU: 사용 불가 (CPU 모드)")
    except ImportError:
        print("GPU: PyTorch 미설치")
    
    # 웹캠 상태
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("웹캠: 사용 가능")
            cap.release()
        else:
            print("웹캠: 사용 불가")
    except:
        print("웹캠: 확인 실패")
    
    # 마이크 상태
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if input_devices:
            print(f"마이크: 사용 가능 ({len(input_devices)}개 장치)")
        else:
            print("마이크: 사용 불가")
    except:
        print("마이크: 확인 실패")
    
    print("=" * 40)

def main():
    """메인 함수"""
    while True:
        show_menu()
        choice = input("\n선택하세요 (0-5): ").strip()
        
        if choice == "0":
            print("👋 프로그램을 종료합니다.")
            break
        elif choice == "1":
            check_dependencies()
            check_models()
            run_main()
        elif choice == "2":
            print("\n🤖 얼굴 감정 분석 테스트를 시작합니다...")
            run_individual_test("face")
        elif choice == "3":
            print("\n🎤 음성 감정 분석 테스트를 시작합니다...")
            run_individual_test("voice")
        elif choice == "4":
            print("\n📝 텍스트 감정 분석 테스트를 시작합니다...")
            run_individual_test("text")
        elif choice == "5":
            check_system_status()
        else:
            print("❌ 잘못된 선택입니다. 다시 시도해주세요.")
        
        input("\n계속하려면 엔터를 누르세요...")
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
