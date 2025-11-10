# 🎤 음성 입력 기능 업데이트

**업데이트 날짜:** 2025년 11월 10일

---

## ✨ 새로운 기능

### 1. **음성 입력 추가** 🎤

- ✅ 마이크 버튼 클릭으로 음성 녹음
- ✅ OpenAI Whisper API로 STT (Speech-to-Text)
- ✅ 음성 → 텍스트 → ChatGPT 응답
- ✅ 사용자 음성은 **터미널에만** 표시
- ✅ AI 응답은 **채팅창에만** 표시

### 2. **화면 비율 개선** 📐

- ✅ 왼쪽(감정 분석): 오른쪽(채팅) = **1 : 1.5**
- ✅ 채팅 영역이 더 넓어져 대화하기 편리

### 3. **실시간 감정 주기 단축** ⚡

- ✅ Late Fusion 간격: **5초 → 2초**
- ✅ 더 빠른 감정 업데이트

---

## 🎯 사용 방법

### 음성으로 대화하기

1. **🎤 버튼 클릭** (분홍색 마이크 버튼)
2. **말하기** (녹음 중... 표시)
3. **🎤 버튼 다시 클릭** (녹음 종료)
4. **자동 처리**:
   - 음성 → 텍스트 변환
   - 텍스트 → ChatGPT 전송
   - AI 응답 채팅창에 표시

**터미널 로그 예시:**
```
INFO - 🎤 [STT 변환] 오늘 기분이 좋지 않아요
INFO - 🎤 [음성 입력] 오늘 기분이 좋지 않아요
```

**채팅창:**
```
AI: 기분이 좋지 않으시다니 안타깝습니다. 
    무슨 일이 있으셨나요? 💙
```

### 텍스트로 대화하기 (기존)

1. 입력창에 텍스트 입력
2. **전송** 버튼 클릭 또는 **Shift+Enter**
3. 사용자 메시지와 AI 응답 모두 채팅창에 표시

---

## 🔧 기술 구현

### 백엔드 (app.py)

#### 1. STT API 엔드포인트

```python
@app.route('/api/voice/transcribe', methods=['POST'])
def transcribe_voice():
    """음성을 텍스트로 변환 (STT)"""
    audio_file = request.files['audio']
    
    # Whisper API 호출
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="ko"
    )
    
    text = transcript.text.strip()
    logger.info(f"🎤 [STT 변환] {text}")
    
    return jsonify({"text": text})
```

#### 2. 음성 입력 구분

```python
@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = data.get('message', '').strip()
    is_voice = data.get('is_voice', False)
    
    # 음성 입력인 경우 터미널에만 로그
    if is_voice:
        logger.info(f"🎤 [음성 입력] {user_message}")
    
    # ChatGPT 응답 생성
    ai_response = get_chatgpt_response(user_message, current_emotions)
    
    return jsonify({"message": ai_response, "is_voice": is_voice})
```

### 프론트엔드

#### 1. 음성 녹음 (app.js)

```javascript
async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    
    mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        await sendVoiceMessage(audioBlob);
    };
    
    mediaRecorder.start();
    isRecording = true;
}
```

#### 2. 음성 전송 및 처리

```javascript
async function sendVoiceMessage(audioBlob) {
    // 1. STT: 음성 → 텍스트
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');
    
    const sttResponse = await fetch('/api/voice/transcribe', {
        method: 'POST',
        body: formData
    });
    
    const { text } = await sttResponse.json();
    
    // 2. ChatGPT: 텍스트 → AI 응답
    const chatResponse = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            message: text,
            is_voice: true  // 음성 입력 플래그
        })
    });
    
    const { message } = await chatResponse.json();
    
    // 3. AI 응답만 채팅창에 표시
    addMessageToChat('ai', message);
}
```

#### 3. UI 업데이트

```javascript
// 녹음 시작
voiceButton.classList.add('recording');  // 빨간색으로 변경
voiceStatus.style.display = 'flex';      // "녹음 중..." 표시

// 녹음 종료
voiceButton.classList.remove('recording');
voiceStatus.style.display = 'none';
```

---

## 🎨 UI 변경사항

### 1. 마이크 버튼

```html
<button id="voice-button" class="voice-button">
    <span class="icon">🎤</span>
</button>
```

**스타일:**
- 평상시: 분홍색 그라디언트
- 녹음 중: 빨간색 + 깜빡임 효과

### 2. 녹음 상태 표시

```html
<div id="voice-status" class="voice-status">
    <span class="recording-dot"></span>
    <span>녹음 중... (말씀하세요)</span>
</div>
```

### 3. 화면 비율

```css
.main-layout {
    grid-template-columns: 1fr 1.5fr;  /* 1 : 1.5 비율 */
}
```

---

## 📊 업데이트 요약

| 항목 | 이전 | 현재 |
|------|------|------|
| **입력 방식** | 텍스트만 | 텍스트 + 음성 |
| **음성 처리** | 없음 | Whisper STT |
| **사용자 음성 표시** | - | 터미널에만 |
| **AI 응답 표시** | 채팅창 | 채팅창 |
| **화면 비율** | 1 : 1.2 | 1 : 1.5 |
| **감정 업데이트** | 5초 | 2초 |

---

## 🚀 실행 방법

### 1. 서버 실행

```bash
cd /Users/qkrwnsmir/Desktop/seeksick-main
python3 app.py
```

### 2. 브라우저 접속

```
http://localhost:5001
```

### 3. 마이크 권한 허용

처음 🎤 버튼 클릭 시 브라우저에서 마이크 권한 요청:
- **허용** 클릭

---

## 💡 사용 팁

### 음성 입력

1. **짧게 말하기**: 3-10초 정도가 적당
2. **명확하게 발음**: 인식률 향상
3. **조용한 환경**: 배경 소음 최소화

### 텍스트 입력

- **Shift+Enter**: 빠른 전송
- **긴 문장도 가능**: 제한 없음

### 혼합 사용

- 음성과 텍스트를 번갈아 사용 가능
- 상황에 맞게 선택

---

## 🔍 터미널 로그 예시

### 음성 입력 시

```
2025-11-10 19:30:15,123 - INFO - 🎤 [STT 변환] 오늘 날씨가 좋네요
2025-11-10 19:30:15,234 - INFO - 🎤 [음성 입력] 오늘 날씨가 좋네요
2025-11-10 19:30:17,456 - INFO - 📊 감정 업데이트: happy (68.0%)
```

### 텍스트 입력 시

```
2025-11-10 19:31:20,789 - INFO - 📊 감정 업데이트: neutral (45.0%)
```

**차이점:**
- 음성: `🎤 [음성 입력]` 로그 표시
- 텍스트: 별도 로그 없음 (채팅창에만 표시)

---

## ⚠️ 주의사항

### 1. 마이크 권한

- 브라우저에서 마이크 권한 필요
- HTTPS 또는 localhost에서만 작동

### 2. 음성 인식 정확도

- 한국어 지원 (Whisper API)
- 명확한 발음 권장
- 배경 소음 주의

### 3. API 사용량

- Whisper API: 음성 1분당 $0.006
- ChatGPT API: 토큰당 과금
- 적절히 사용 권장

---

## 🎉 완료!

**모든 기능이 업데이트되었습니다!**

### 주요 개선사항

✅ 음성 입력 추가 (🎤 버튼)  
✅ 사용자 음성은 터미널에만  
✅ AI 응답은 채팅창에만  
✅ 화면 비율 개선 (1:1.5)  
✅ 감정 업데이트 2초로 단축  

---

**지금 실행하고 음성으로 대화해보세요!** 🚀

```bash
python3 app.py
# http://localhost:5001
```

