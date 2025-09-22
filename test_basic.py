#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
기본 기능 테스트 스크립트
모델 파일 없이도 기본 구조와 라이브러리 동작을 확인합니다.
"""

import sys
import traceback

def test_imports():
    """필수 라이브러리 임포트 테스트"""
    print("📦 라이브러리 임포트 테스트...")
    
    libraries = [
        ("numpy", "수치 계산"),
        ("cv2", "OpenCV 영상 처리"),
        ("torch", "PyTorch 딥러닝"),
        ("transformers", "Hugging Face Transformers"),
        ("sounddevice", "음성 녹음"),
        ("scipy", "과학 계산"),
        ("whisper", "OpenAI Whisper"),
    ]
    
    failed_imports = []
    
    for lib_name, description in libraries:
        try:
            if lib_name == "cv2":
                import cv2
            elif lib_name == "torch":
                import torch
            elif lib_name == "transformers":
                import transformers
            elif lib_name == "sounddevice":
                import sounddevice
            elif lib_name == "scipy":
                import scipy
            elif lib_name == "whisper":
                import whisper
            elif lib_name == "numpy":
                import numpy
            
            print(f"  ✅ {lib_name}: {description}")
            
        except ImportError as e:
            print(f"  ❌ {lib_name}: {description} - {e}")
            failed_imports.append(lib_name)
    
    return failed_imports

def test_device_access():
    """디바이스 접근 테스트"""
    print("\n🎥 디바이스 접근 테스트...")
    
    # 웹캠 테스트
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"  ✅ 웹캠: 접근 가능 (해상도: {frame.shape[1]}x{frame.shape[0]})")
            else:
                print("  ⚠️ 웹캠: 열렸지만 프레임 읽기 실패")
            cap.release()
        else:
            print("  ❌ 웹캠: 접근 불가")
    except Exception as e:
        print(f"  ❌ 웹캠: 테스트 실패 - {e}")
    
    # 마이크 테스트
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        
        if input_devices:
            default_input = sd.default.device[0]
            device_name = devices[default_input]['name']
            print(f"  ✅ 마이크: 접근 가능 (기본: {device_name})")
            print(f"      총 {len(input_devices)}개 입력 장치 발견")
        else:
            print("  ❌ 마이크: 입력 장치를 찾을 수 없음")
            
    except Exception as e:
        print(f"  ❌ 마이크: 테스트 실패 - {e}")

def test_torch_setup():
    """PyTorch 설정 테스트"""
    print("\n🔥 PyTorch 설정 테스트...")
    
    try:
        import torch
        print(f"  ✅ PyTorch 버전: {torch.__version__}")
        
        # CUDA 확인
        if torch.cuda.is_available():
            print(f"  ✅ CUDA: 사용 가능 (버전: {torch.version.cuda})")
            print(f"      GPU 개수: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"      GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("  ⚠️ CUDA: 사용 불가 (CPU 모드로 동작)")
        
        # 간단한 텐서 연산 테스트
        x = torch.rand(3, 3)
        y = torch.rand(3, 3)
        z = torch.mm(x, y)
        print(f"  ✅ 텐서 연산: 정상 동작")
        
    except Exception as e:
        print(f"  ❌ PyTorch: 테스트 실패 - {e}")

def test_model_structure():
    """모델 구조 테스트 (파일 없이)"""
    print("\n🏗️ 모델 구조 테스트...")
    
    try:
        # 얼굴 감정 모델
        from models.face_emotion_model import FaceEmotionResNet18
        face_model = FaceEmotionResNet18(num_emotions=5)
        print("  ✅ 얼굴 감정 모델: 구조 정상")
        
        # 음성 감정 모델
        from models.voice_emotion_model import VoiceEmotionCNN
        voice_model = VoiceEmotionCNN(input_dim=128, num_emotions=5)
        print("  ✅ 음성 감정 모델: 구조 정상")
        
        # 텍스트 감정 모델
        from models.text_emotion_model import KoBERTForEmotion
        # KoBERT는 인터넷 연결이 필요하므로 간단히 클래스만 확인
        print("  ✅ 텍스트 감정 모델: 클래스 정상")
        
    except Exception as e:
        print(f"  ❌ 모델 구조: 테스트 실패 - {e}")
        traceback.print_exc()

def test_whisper():
    """Whisper 모델 테스트"""
    print("\n🎙️ Whisper 모델 테스트...")
    
    try:
        import whisper
        
        # 가장 작은 모델로 테스트
        print("  🔄 Whisper tiny 모델 로드 중...")
        model = whisper.load_model("tiny")
        print("  ✅ Whisper: 모델 로드 성공")
        
        # 더미 오디오로 테스트
        import numpy as np
        dummy_audio = np.random.random(16000).astype(np.float32)  # 1초 더미 오디오
        result = model.transcribe(dummy_audio)
        print(f"  ✅ Whisper: 추론 테스트 완료")
        
    except Exception as e:
        print(f"  ⚠️ Whisper: 테스트 실패 - {e}")
        print("      인터넷 연결이나 모델 다운로드 문제일 수 있습니다.")

def main():
    """메인 테스트 함수"""
    print("🧪 Seeksick 멀티모달 감정 분석 기본 테스트")
    print("=" * 60)
    
    # 1. 라이브러리 임포트 테스트
    failed_imports = test_imports()
    
    if failed_imports:
        print(f"\n❌ {len(failed_imports)}개 라이브러리 설치 필요:")
        print("다음 명령어로 설치하세요:")
        print("pip install " + " ".join(failed_imports))
        print("\n또는:")
        print("python setup.py")
        return
    
    # 2. 디바이스 접근 테스트
    test_device_access()
    
    # 3. PyTorch 설정 테스트
    test_torch_setup()
    
    # 4. 모델 구조 테스트
    test_model_structure()
    
    # 5. Whisper 테스트
    test_whisper()
    
    print("\n" + "=" * 60)
    print("✅ 기본 테스트 완료!")
    print("\n다음 단계:")
    print("1. 모델 파일들을 models/ 디렉토리에 배치")
    print("   - seeksick-resnet18.pth")
    print("   - seeksick-voice.pt") 
    print("   - seeksick-kobert.pt")
    print("2. 메인 프로그램 실행:")
    print("   python main.py")
    print("   또는")
    print("   python run.py")

if __name__ == "__main__":
    main()
