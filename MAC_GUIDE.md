# 🍎 맥북에서 Seeksick 실행 가이드

## 📱 아이폰이 아니라 맥북에서 실행해야 합니다!

이 프로그램은 **맥북의 웹캠과 마이크**를 사용하므로, 맥북에서 직접 실행해야 합니다.

---

## 🖥️ 현재 상황

현재 터미널/SSH 환경에서는:
- ✅ **텍스트 감정 분석**: 잘 작동 (카메라/마이크 불필요)
- ❌ **얼굴 감정 분석**: 웹캠 필요 → 맥북에서 직접 실행 필요
- ❌ **음성 감정 분석**: 마이크 필요 → 맥북에서 직접 실행 필요

---

## ✅ 해결 방법 3가지

### 방법 1: 텍스트 감정 분석만 사용 (현재 환경에서 가능) ⭐

**웹캠/마이크 없이도 사용 가능!**

```bash
# 터미널에서 바로 실행
cd /Users/qkrwnsmir/Desktop/seeksick-main
python3 text_only_demo.py
```

**또는 대화형 모드:**

```bash
python3 test_realtime.py
```

직접 문장을 입력하면 실시간으로 감정을 분석해줍니다!

---

### 방법 2: 맥북에서 직접 실행 (전체 기능 사용) 🎥

맥북에서 **Terminal.app**을 열고:

```bash
# 1. 프로젝트 폴더로 이동
cd /Users/qkrwnsmir/Desktop/seeksick-main

# 2. 전체 멀티모달 시스템 실행
python3 main.py
```

**실행되면:**
- 웹캠이 자동으로 켜집니다 🎥
- 마이크가 녹음을 시작합니다 🎤
- 3초마다 음성/텍스트 감정 분석
- 실시간 얼굴 감정 분석
- 종료: `q` 키

**⚠️ 처음 실행 시 권한 요청:**
- "카메라 접근 허용" → **허용**
- "마이크 접근 허용" → **허용**

---

### 방법 3: 개별 모델 테스트

맥북 Terminal에서:

```bash
cd /Users/qkrwnsmir/Desktop/seeksick-main

# 얼굴 감정만 (웹캠 필요)
python3 -m models.face_emotion_model

# 음성 감정만 (마이크 필요)
python3 -m models.voice_emotion_model

# 텍스트 감정만 (장비 불필요)
python3 -m models.text_emotion_model
```

---

## 🎯 추천 실행 순서

### 1️⃣ 먼저 텍스트 분석 테스트 (지금 바로 가능)

```bash
cd /Users/qkrwnsmir/Desktop/seeksick-main
python3 text_only_demo.py
```

**결과:**
- 8개 예문 자동 분석
- 이모지와 함께 결과 출력
- 웹캠/마이크 불필요 ✅

---

### 2️⃣ 대화형 텍스트 분석 (지금 바로 가능)

```bash
python3 test_realtime.py
```

**사용법:**
```
💭 문장 입력: 오늘 정말 기분이 좋아요!

   😊 예측 감정: HAPPY
   📊 신뢰도: 99.0%
   
   🏆 상위 3개 감정:
      1위: 😊 happy       99.0%
      2위: 😢 depressed   0.3%
      3위: 😐 neutral     0.2%
```

종료: `quit` 또는 `종료` 입력

---

### 3️⃣ 맥북에서 전체 시스템 실행

맥북 화면 앞에서:

```bash
python3 main.py
```

**화면에 나타나는 것:**
- 웹캠 영상 + 얼굴 감정 표시
- 음성 감정 분석 결과
- 텍스트 감정 분석 결과
- 실시간 업데이트

---

## 📂 파일 위치

프로젝트가 설치된 위치:

```
/Users/qkrwnsmir/Desktop/seeksick-main/
```

맥북 Finder에서:
1. **Finder** 열기
2. **이동** → **홈** (Cmd+Shift+H)
3. **Desktop** → **seeksick-main** 폴더

---

## 🎥 맥북에서 실행하는 방법 (단계별)

### 맥북에서 Terminal 열기

1. **방법 1**: Spotlight 검색
   - `Cmd + Space` 누르기
   - "Terminal" 입력
   - Enter

2. **방법 2**: Launchpad
   - **Launchpad** 열기
   - **기타** 폴더
   - **터미널** 클릭

3. **방법 3**: Finder
   - **응용프로그램** → **유틸리티** → **터미널**

### 실행 명령어

```bash
# 프로젝트 폴더로 이동
cd Desktop/seeksick-main

# 텍스트 분석만 (웹캠/마이크 불필요)
python3 text_only_demo.py

# 또는 전체 시스템 (웹캠/마이크 필요)
python3 main.py
```

---

## ⚠️ 권한 설정 (맥북에서 처음 실행 시)

### 카메라 권한

**macOS가 물어보면:**
```
"Python"에서 카메라에 접근하려고 합니다.
```

→ **허용** 클릭

**수동 설정:**
1. **시스템 설정** (System Settings)
2. **개인정보 보호 및 보안** (Privacy & Security)
3. **카메라** (Camera)
4. **Python** 또는 **Terminal** 체크

### 마이크 권한

**macOS가 물어보면:**
```
"Python"에서 마이크에 접근하려고 합니다.
```

→ **허용** 클릭

**수동 설정:**
1. **시스템 설정**
2. **개인정보 보호 및 보안**
3. **마이크** (Microphone)
4. **Python** 또는 **Terminal** 체크

---

## 🎬 실행 예시

### 현재 터미널에서 (텍스트만)

```bash
cd /Users/qkrwnsmir/Desktop/seeksick-main
python3 text_only_demo.py
```

**출력 예시:**
```
😊 긍정적인 문장
💬 문장: '오늘 정말 기분이 좋아요!'

   😊 예측 감정: HAPPY
   📊 신뢰도: 99.1%

   🏆 상위 3개 감정:
      1위: 😊 happy        99.1%
      2위: 😢 depressed    0.3%
      3위: 😐 neutral      0.2%
```

### 맥북에서 직접 (전체 기능)

```bash
python3 main.py
```

**화면 출력:**
```
멀티모달 감정 분석을 시작합니다...
종료하려면 'q'를 누르세요.

[웹캠 창이 열림]
👤 얼굴 감정 분석 (14:23:15):
  happy: 0.734
  depressed: 0.056
  surprised: 0.145

🎤 음성 감정 분석:
  happy: 0.782
  depressed: 0.045

📝 텍스트 감정 분석:
  happy: 0.856
  depressed: 0.024
```

---

## 💡 팁

### 빠른 테스트

현재 환경에서 바로 테스트하려면:

```bash
cd /Users/qkrwnsmir/Desktop/seeksick-main

# 방법 1: 자동 데모 (8개 예문)
python3 text_only_demo.py

# 방법 2: 직접 입력
python3 test_realtime.py

# 방법 3: 빠른 테스트
python3 -c "
from models.text_emotion_model import TextEmotionAnalyzer
analyzer = TextEmotionAnalyzer()
text = '오늘 정말 행복해요!'
probs = analyzer.analyze_text(text)
emotion, conf = analyzer.get_emotion_label(probs)
print(f'감정: {emotion} ({conf:.1%})')
"
```

### 파일 확인

```bash
# 사용 가능한 프로그램 목록
ls -lh *.py

# 출력:
# main.py              - 전체 멀티모달 시스템
# test_realtime.py     - 대화형 텍스트 분석
# text_only_demo.py    - 자동 텍스트 데모
# demo_analysis.py     - 통합 테스트
# run.py              - 메뉴 기반 실행
```

---

## 📊 요약

| 프로그램 | 웹캠 | 마이크 | 실행 위치 |
|---------|------|--------|-----------|
| `text_only_demo.py` | ❌ | ❌ | 지금 바로 실행 가능! |
| `test_realtime.py` | ❌ | ❌ | 지금 바로 실행 가능! |
| `demo_analysis.py` | ❌ | ❌ | 지금 바로 실행 가능! |
| `main.py` | ✅ | ✅ | 맥북에서 직접 실행 필요 |
| 얼굴 감정 테스트 | ✅ | ❌ | 맥북에서 직접 실행 필요 |
| 음성 감정 테스트 | ❌ | ✅ | 맥북에서 직접 실행 필요 |

---

## 🎉 결론

### 지금 당장 사용하려면:

```bash
cd /Users/qkrwnsmir/Desktop/seeksick-main
python3 text_only_demo.py
```

웹캠/마이크 없이도 **텍스트 감정 분석**은 완벽하게 작동합니다!

### 전체 기능을 사용하려면:

**맥북 화면 앞에서** Terminal을 열고:

```bash
cd /Users/qkrwnsmir/Desktop/seeksick-main
python3 main.py
```

웹캠과 마이크를 사용한 **실시간 멀티모달 감정 분석**이 가능합니다!

---

**💡 추천:** 먼저 `text_only_demo.py`로 텍스트 분석을 테스트해보고, 맥북에서 `main.py`를 실행해보세요!

