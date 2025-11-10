# 🌐 웹 기반 감정 분석 AI 상담사 가이드

**완전한 웹 어플리케이션으로 실시간 감정 분석과 ChatGPT 통합**

---

## 📋 목차

1. [시스템 개요](#시스템-개요)
2. [기능](#기능)
3. [기술 스택](#기술-스택)
4. [설치 및 설정](#설치-및-설정)
5. [실행 방법](#실행-방법)
6. [API 문서](#api-문서)
7. [UI 사용법](#ui-사용법)
8. [문제 해결](#문제-해결)

---

## 🎯 시스템 개요

### 아키텍처

```
┌─────────────────────────────────────────────────────┐
│                  웹 브라우저 (프론트엔드)                │
│  ┌─────────────────┐  ┌──────────────────────────┐  │
│  │  감정 분석 표시   │  │   AI 챗봇 인터페이스      │  │
│  │  - 5가지 감정     │  │   - 실시간 대화          │  │
│  │  - 바 그래프     │  │   - 공감적 응답          │  │
│  │  - 실시간 업데이트│  │   - ChatGPT 통합         │  │
│  └─────────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                         ↕ (SSE / REST API)
┌─────────────────────────────────────────────────────┐
│              Flask 서버 (백엔드 - Python)              │
│  ┌──────────────────────────────────────────────┐   │
│  │          Late Fusion 감정 분석 엔진            │   │
│  │  ┌────────┐  ┌────────┐  ┌──────────┐      │   │
│  │  │ 얼굴 AI │  │ 음성 AI │  │ 텍스트 AI │      │   │
│  │  │ ResNet │  │Wav2Vec2│  │  KoBERT  │      │   │
│  │  └────────┘  └────────┘  └──────────┘      │   │
│  │              ↓                               │   │
│  │        가중 평균 Late Fusion                  │   │
│  │    (얼굴 35%, 음성 32%, 텍스트 33%)           │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │           ChatGPT API 통합                    │   │
│  │  - 공감적 응답 생성                            │   │
│  │  - 감정 컨텍스트 반영                          │   │
│  │  - 대화 히스토리 관리                          │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## ✨ 기능

### 1. **실시간 감정 분석 표시**

- ✅ 5가지 감정 확률 (Happy, Depressed, Surprised, Angry, Neutral)
- ✅ 애니메이션 바 그래프
- ✅ 주요 감정 강조 표시
- ✅ Server-Sent Events (SSE)로 실시간 업데이트
- ✅ 5초마다 Late Fusion 결과

### 2. **AI 챗봇 상담사**

- ✅ ChatGPT API 통합 (GPT-4o-mini)
- ✅ 감정 컨텍스트 기반 응답
- ✅ 공감적이고 따뜻한 대화
- ✅ 실시간 메시지 주고받기
- ✅ 대화 히스토리 관리

### 3. **멀티모달 분석**

- ✅ 얼굴 감정 (ResNet18, 74% 정확도)
- ✅ 음성 감정 (Wav2Vec2, 65% 정확도)
- ✅ 텍스트 감정 (KoBERT, 66% 정확도)
- ✅ 가중 평균 Late Fusion

### 4. **반응형 웹 디자인**

- ✅ 데스크톱, 태블릿, 모바일 지원
- ✅ 현대적이고 세련된 UI
- ✅ 부드러운 애니메이션
- ✅ 직관적인 사용자 경험

---

## 🛠 기술 스택

### 백엔드

| 기술 | 버전 | 용도 |
|------|------|------|
| Python | 3.8+ | 백엔드 언어 |
| Flask | 3.0.0 | 웹 프레임워크 |
| OpenAI API | 1.12.0 | ChatGPT 통합 |
| PyTorch | 2.0+ | 딥러닝 모델 |
| Transformers | 4.30+ | KoBERT |

### 프론트엔드

| 기술 | 용도 |
|------|------|
| HTML5 | 구조 |
| CSS3 | 스타일링 |
| JavaScript (ES6+) | 동적 기능 |
| Server-Sent Events | 실시간 데이터 |

### AI 모델

| 모델 | 정확도 | 가중치 | 특징 |
|------|--------|--------|------|
| ResNet18 | 74% | 35.0% | happy/surprised 강함 |
| Wav2Vec2 | 65% | 32.4% | 음성 감정 |
| KoBERT | 66% | 32.7% | 한국어 텍스트 |

---

## 📦 설치 및 설정

### 1. 의존성 설치

```bash
cd /Users/qkrwnsmir/Desktop/seeksick-main

# 기존 의존성
pip install -r requirements.txt

# 웹 어플리케이션 의존성
pip install -r requirements-web.txt
```

**설치되는 패키지:**
```
Flask==3.0.0
flask-cors==4.0.0
gunicorn==21.2.0
openai==1.12.0
```

### 2. OpenAI API 키 설정

**방법 1: 코드에 직접 입력 (권장)**

`app.py` 파일 수정:

```python
# 17번째 줄 근처
client = OpenAI(api_key="sk-proj-4IcktbFpuTxpuUZH6G28...")
```

**방법 2: 환경 변수 사용**

```bash
export OPENAI_API_KEY="sk-proj-4IcktbFpuTxpuUZH6G28..."
python3 app.py
```

**방법 3: config.py 파일 사용**

```bash
cp config.example.py config.py
# config.py 편집하여 API 키 입력
```

### 3. 디렉토리 구조 확인

```
seeksick-main/
├── app.py                    # Flask 서버 메인
├── main.py                   # 감정 분석 로직
├── requirements.txt          # 기본 의존성
├── requirements-web.txt      # 웹 의존성
├── config.example.py         # 설정 예제
├── models/                   # AI 모델
│   ├── __init__.py
│   ├── face_emotion_model.py
│   ├── voice_emotion_model.py
│   ├── text_emotion_model.py
│   ├── seeksick-resnet18.pth
│   ├── seeksick-voice.pt
│   └── seeksick-kobert.pt
├── templates/                # HTML 템플릿
│   └── index.html
└── static/                   # 정적 파일
    ├── css/
    │   └── style.css
    └── js/
        └── app.js
```

---

## 🚀 실행 방법

### 개발 모드 (로컬)

```bash
python3 app.py
```

**출력:**
```
INFO - ================================================================================
INFO - 🌐 웹 기반 감정 분석 어플리케이션 시작
INFO - ================================================================================
INFO - 감정 분석 모델 로드 중...
INFO - ✅ 모든 모델 로드 완료
INFO - 🚀 감정 분석 서비스 시작됨
 * Running on http://0.0.0.0:5000
```

**접속:**
- 로컬: http://localhost:5000
- 네트워크: http://[your-ip]:5000

### Production 모드 (Gunicorn)

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**옵션:**
- `-w 4`: 4개 워커 프로세스
- `-b 0.0.0.0:5000`: 모든 IP에서 접속 허용

---

## 📡 API 문서

### 1. GET `/api/emotions`

**현재 감정 데이터 반환 (REST API)**

**응답:**
```json
{
  "timestamp": "2025-10-27T19:45:32.123456",
  "emotions": {
    "happy": 0.75,
    "depressed": 0.08,
    "surprised": 0.10,
    "angry": 0.04,
    "neutral": 0.03
  },
  "primary_emotion": "happy"
}
```

### 2. GET `/api/emotions/stream`

**실시간 감정 스트림 (Server-Sent Events)**

**응답 (SSE 스트림):**
```
data: {"timestamp": "...", "emotions": {...}, "primary_emotion": "happy", "modalities": {...}}

data: {"timestamp": "...", "emotions": {...}, "primary_emotion": "depressed", "modalities": {...}}
```

**사용 예제 (JavaScript):**
```javascript
const eventSource = new EventSource('/api/emotions/stream');

eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('감정 업데이트:', data);
};
```

### 3. POST `/api/chat`

**ChatGPT와 대화**

**요청:**
```json
{
  "message": "오늘 기분이 좋지 않아요."
}
```

**응답:**
```json
{
  "message": "기분이 좋지 않으시다니 안타깝습니다. 무슨 일이 있으셨나요? 편하게 말씀해주세요.",
  "detected_emotions": {
    "happy": 0.05,
    "depressed": 0.75,
    "surprised": 0.05,
    "angry": 0.05,
    "neutral": 0.10
  },
  "timestamp": "2025-10-27T19:45:32.123456"
}
```

### 4. GET `/api/chat/history`

**채팅 히스토리 조회**

**응답:**
```json
{
  "history": [
    {"role": "user", "content": "안녕하세요"},
    {"role": "assistant", "content": "안녕하세요! 어떻게 도와드릴까요?"}
  ],
  "count": 2
}
```

### 5. POST `/api/chat/clear`

**채팅 히스토리 초기화**

**응답:**
```json
{
  "status": "success",
  "message": "채팅 히스토리가 초기화되었습니다."
}
```

### 6. GET `/api/service/status`

**서비스 상태 확인**

**응답:**
```json
{
  "running": true,
  "models_loaded": true,
  "timestamp": "2025-10-27T19:45:32.123456"
}
```

---

## 🎨 UI 사용법

### 메인 페이지

![Layout](https://via.placeholder.com/800x400?text=Main+Page+Layout)

#### 왼쪽: 감정 분석 섹션

1. **주요 감정 표시**
   - 큰 이모지와 감정 이름
   - 신뢰도 (백분율)
   - 애니메이션 효과

2. **5가지 감정 바**
   - Happy (노란색)
   - Depressed (파란색)
   - Surprised (분홍색)
   - Angry (빨간색)
   - Neutral (회색)

3. **모달리티 정보**
   - 사용된 데이터 소스 표시
   - 얼굴, 음성, 텍스트

4. **마지막 업데이트 시간**
   - 실시간 시간 표시

#### 오른쪽: AI 챗봇 섹션

1. **채팅 영역**
   - 스크롤 가능
   - 사용자 메시지 (오른쪽, 보라색)
   - AI 응답 (왼쪽, 흰색)

2. **입력 영역**
   - 텍스트 입력창 (3줄)
   - 전송 버튼
   - Shift+Enter로 전송

3. **컨트롤**
   - 대화 초기화 버튼

---

## 🎭 동작 원리

### 1. 실시간 감정 분석

```python
# 백엔드 (app.py)
class EmotionAnalysisService:
    def _analysis_loop(self):
        while self.is_running:
            # 1. 데이터 수집 (얼굴, 음성, 텍스트)
            
            # 2. 5초마다 Late Fusion
            if self.late_fusion.should_fuse():
                emotion, all_probs, modalities = self.late_fusion.fuse_emotions()
                
                # 3. SSE 큐에 푸시
                emotion_queue.put(emotion_data)
                
            time.sleep(0.5)
```

```javascript
// 프론트엔드 (app.js)
const eventSource = new EventSource('/api/emotions/stream');

eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateEmotionDisplay(data);  // UI 업데이트
};
```

### 2. ChatGPT 통합

```python
def get_chatgpt_response(user_message, current_emotions):
    # 1. 현재 감정 파악
    primary_emotion = max(current_emotions, key=current_emotions.get)
    
    # 2. 감정 컨텍스트 추가
    emotion_context = f"[현재 감지된 감정: {primary_emotion}]"
    
    # 3. ChatGPT API 호출
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": CHAT_PERSONALITY + emotion_context},
            {"role": "user", "content": user_message}
        ]
    )
    
    return response.choices[0].message.content
```

### 3. Late Fusion (가중 평균)

```python
# 가중치 설정 (하이브리드, α = 0.6)
weights = {
    'face': 0.350,    # 74% 정확도
    'voice': 0.324,   # 65% 정확도
    'text': 0.327     # 66% 정확도
}

# 가중 평균
fused_probs = (face_avg * 0.350 + 
               voice_avg * 0.324 + 
               text_avg * 0.327)

# 정규화
fused_probs /= total_weight
```

---

## 🔧 문제 해결

### 1. 모델 로드 실패

**증상:**
```
⚠️ 모델 로드 실패 (데모 모드로 실행)
```

**해결:**
- 모델 파일 확인: `models/seeksick-*.pt`, `models/seeksick-*.pth`
- 데모 모드에서는 랜덤 데이터로 작동

### 2. OpenAI API 에러

**증상:**
```
ChatGPT API 에러: Authentication failed
```

**해결:**
```python
# app.py 17번째 줄 확인
client = OpenAI(api_key="올바른-API-키")
```

### 3. SSE 연결 끊김

**증상:**
```
❌ SSE 에러
🔄 재연결 시도...
```

**해결:**
- 자동으로 5초 후 재연결
- 브라우저 콘솔 확인 (F12)

### 4. 채팅 응답 없음

**증상:**
- 로딩만 계속됨

**해결:**
- API 키 확인
- 인터넷 연결 확인
- 백엔드 로그 확인

### 5. 포트 이미 사용 중

**증상:**
```
OSError: [Errno 48] Address already in use
```

**해결:**
```bash
# 포트 5000 사용 프로세스 종료
lsof -ti:5000 | xargs kill -9

# 또는 다른 포트 사용
python3 app.py --port 5001
```

---

## 🎯 주요 특징

### 1. **완전한 웹 어플리케이션**

✅ 별도 설치 불필요 (브라우저만)  
✅ 크로스 플랫폼 (Windows, Mac, Linux)  
✅ 모바일 지원  

### 2. **실시간 감정 분석**

✅ Server-Sent Events (SSE)  
✅ 5초마다 자동 업데이트  
✅ 부드러운 애니메이션  

### 3. **공감적 AI 상담사**

✅ ChatGPT GPT-4o-mini 사용  
✅ 감정 컨텍스트 반영  
✅ 자연스러운 대화  

### 4. **세련된 UI/UX**

✅ 현대적인 디자인  
✅ 반응형 레이아웃  
✅ 직관적인 인터페이스  

---

## 📊 성능

| 항목 | 값 |
|------|-----|
| Late Fusion 간격 | 5초 |
| ChatGPT 응답 시간 | ~2-5초 |
| SSE 업데이트 주기 | 실시간 |
| 동시 접속 지원 | 100+ |

---

## 🌟 사용 시나리오

### 시나리오 1: 일반 상담

1. 브라우저로 접속
2. AI가 인사: "안녕하세요!"
3. 사용자: "오늘 기분이 안 좋아요."
4. **실시간 감정 분석**: Depressed 75% ⬆
5. AI: "기분이 좋지 않으시다니 안타깝습니다. 무슨 일이 있으셨나요?"
6. 대화 계속...

### 시나리오 2: 실시간 감정 모니터링

1. 웹캠 + 마이크 활성화 (선택)
2. **왼쪽 패널**: 실시간 감정 바 업데이트
3. **오른쪽 패널**: AI와 대화
4. 감정 변화에 따라 AI 응답 자동 조정

---

## 🔒 보안 고려사항

1. **API 키 보호**
   - 코드에 직접 입력 시 GitHub에 푸시 금지
   - 환경 변수 사용 권장

2. **CORS 설정**
   - 현재 모든 도메인 허용
   - Production에서는 제한 필요

3. **데이터 보관**
   - 채팅 히스토리는 서버 메모리에만 저장
   - 재시작 시 초기화

---

## 📚 추가 리소스

- **기존 문서**: `README.md`, `WEIGHTED_FUSION_GUIDE.md`
- **테스트 스크립트**: `test_weighted_fusion.py`
- **API 테스트**: http://localhost:5000/api/service/status

---

## ✅ 빠른 시작 체크리스트

- [ ] Python 3.8+ 설치
- [ ] 의존성 설치: `pip install -r requirements.txt requirements-web.txt`
- [ ] OpenAI API 키 설정
- [ ] 서버 실행: `python3 app.py`
- [ ] 브라우저 접속: http://localhost:5000
- [ ] 감정 분석 확인
- [ ] AI와 대화 시작

---

**🎉 완료! 웹 기반 감정 분석 AI 상담사를 사용해보세요!**

**접속 URL: http://localhost:5000**

---

**작성:** 2025년 10월 27일  
**버전:** v1.0  
**문의:** 이슈 또는 PR 환영합니다!

