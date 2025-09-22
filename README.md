# 🎯 Seeksick 멀티모달 감정 분석 프로그램

실시간으로 **얼굴**, **음성**, **텍스트** 3가지 모달리티에서 감정을 분석하는 통합 프로그램입니다.

## 🎭 지원하는 감정

모든 모달리티에서 동일한 5가지 감정을 분류합니다:
- **행복** (happy)
- **우울** (depressed) 
- **놀람** (surprised)
- **화남** (angry)
- **중립** (neutral)

각 감정의 확률 합은 1.0이며, 예시: `[0.1, 0.1, 0.2, 0.5, 0.1]`

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론 (또는 압축 해제)
cd seeksick-main

# 자동 설치 스크립트 실행
python setup.py
```

### 2. 모델 파일 배치

다음 모델 파일들을 `models/` 디렉토리에 배치하세요:

```
models/
├── seeksick-resnet18.pth    # 얼굴 감정 분석
├── seeksick-voice.pt        # 음성 감정 분석
└── seeksick-kobert.pt       # 텍스트 감정 분석
```

### 3. 프로그램 실행

```bash
# 멀티모달 감정 분석 시작
python main.py
```

## 🎮 사용법

### 메인 프로그램

프로그램을 실행하면:

1. **웹캠**이 자동으로 시작됩니다
2. **마이크**에서 실시간으로 음성을 수집합니다
3. **3초 간격**으로 다음 분석을 수행합니다:
   - 음성 → 텍스트 변환 (Whisper)
   - 음성 감정 분석
   - 텍스트 감정 분석
   - 얼굴 감정 분석 (실시간)

4. 결과가 **콘솔**과 **화면**에 표시됩니다

종료: `q` 키를 누르세요

### 개별 모델 테스트

각 모델을 개별적으로 테스트할 수 있습니다:

```bash
# 얼굴 감정 분석 테스트
python -m models.face_emotion_model

# 음성 감정 분석 테스트 
python -m models.voice_emotion_model

# 텍스트 감정 분석 테스트
python -m models.text_emotion_model
```

## 🏗️ 프로젝트 구조

```
seeksick-main/
├── main.py                    # 메인 프로그램
├── setup.py                   # 설치 스크립트
├── requirements.txt           # 의존성 패키지
├── .env                      # 설정 파일
├── models/                   # 모델 모듈
│   ├── __init__.py
│   ├── face_emotion_model.py   # 얼굴 감정 분석
│   ├── voice_emotion_model.py  # 음성 감정 분석
│   ├── text_emotion_model.py   # 텍스트 감정 분석
│   ├── seeksick-resnet18.pth   # 얼굴 모델 파일
│   ├── seeksick-voice.pt       # 음성 모델 파일
│   └── seeksick-kobert.pt      # 텍스트 모델 파일
└── README.md
```

## 🔧 기술 스택

### 핵심 기술
- **얼굴 감정**: ResNet18 + OpenCV
- **음성 감정**: CNN/Wav2Vec2 + librosa
- **텍스트 감정**: KoBERT + Transformers
- **음성 인식**: OpenAI Whisper

### 주요 라이브러리
- **음성 처리**: `sounddevice`, `scipy`, `librosa`
- **영상 처리**: `opencv-python`
- **딥러닝**: `torch`, `transformers`
- **음성 인식**: `openai-whisper`

## ⚙️ 설정

`.env` 파일에서 다음 설정을 조정할 수 있습니다:

```bash
# 모델 파일 경로
FACE_MODEL_PATH=models/seeksick-resnet18.pth
VOICE_MODEL_PATH=models/seeksick-voice.pt
TEXT_MODEL_PATH=models/seeksick-kobert.pt

# 오디오 설정
SAMPLE_RATE=16000
AUDIO_BUFFER_SIZE=3.0

# 비디오 설정
VIDEO_FPS=10

# 신뢰도 임계값
CONFIDENCE_THRESHOLD=0.5
```

## 📊 출력 예시

```
=============================================================
시간: 14:23:15
텍스트: 오늘 정말 기분이 좋아요!
=============================================================
🎤 음성 감정 분석:
  happy: 0.782
  depressed: 0.045
  surprised: 0.123
  angry: 0.032
  neutral: 0.018

📝 텍스트 감정 분석:
  happy: 0.856
  depressed: 0.024
  surprised: 0.089
  angry: 0.015
  neutral: 0.016

👤 얼굴 감정 분석 (14:23:15):
  happy: 0.734
  depressed: 0.056
  surprised: 0.145
  angry: 0.038
  neutral: 0.027
```

## 🚨 문제 해결

### 모델 파일 없음
```
⚠️ seeksick-resnet18.pth 없음 (나중에 추가 필요)
```
→ 해당 모델 파일을 `models/` 디렉토리에 배치하세요

### 웹캠 접근 실패
```
❌ 웹캠을 열 수 없습니다.
```
→ 카메라 권한을 확인하고 다른 프로그램에서 카메라를 사용하고 있지 않은지 확인하세요

### 마이크 접근 실패
```
⚠️ 사용 가능한 마이크를 찾을 수 없습니다.
```
→ 마이크 권한을 확인하고 오디오 장치가 올바르게 연결되어 있는지 확인하세요

### 메모리 부족
```
CUDA out of memory
```
→ `.env`에서 배치 크기를 줄이거나 CPU 모드로 전환하세요

## 🛠️ 커스터마이징

### 새로운 감정 추가

1. `models/__init__.py`에서 `EMOTIONS` 리스트 수정
2. 각 모델의 출력 차원 변경
3. 모델 재훈련 또는 가중치 수정

### 새로운 모달리티 추가

1. `models/` 디렉토리에 새 모델 모듈 생성
2. `main.py`의 `MultimodalEmotionAnalyzer`에 통합
3. UI 및 결과 표시 로직 수정

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 🤝 기여

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다!

---

**Seeksick Team** 💙