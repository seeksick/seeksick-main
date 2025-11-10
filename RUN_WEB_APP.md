# 🚀 웹 어플리케이션 실행 가이드

## ⚡ 빠른 실행

### 1. 서버 실행

```bash
cd /Users/qkrwnsmir/Desktop/seeksick-main
python3 app.py
```

**포트:** 5001 (기본값)

### 2. 브라우저 접속

```
http://localhost:5001
```

---

## 🔧 포트 변경

### 방법 1: 환경 변수 사용

```bash
PORT=8080 python3 app.py
# http://localhost:8080
```

### 방법 2: 코드 수정

`app.py` 389번째 줄:

```python
port = int(os.environ.get('PORT', 5001))  # 원하는 포트 번호
```

---

## ⚠️ 문제 해결

### "Address already in use" 에러

**원인:** 포트가 이미 사용 중입니다.

**해결 방법 1:** 다른 포트 사용
```bash
PORT=5002 python3 app.py
```

**해결 방법 2:** 기존 프로세스 종료
```bash
# 포트 5001 사용 중인 프로세스 확인
lsof -i :5001

# 프로세스 종료
kill -9 <PID>
```

**해결 방법 3 (macOS):** AirPlay Receiver 비활성화
1. 시스템 설정 (System Preferences)
2. 일반 (General)
3. AirDrop 및 Handoff
4. "AirPlay Receiver" 끄기

---

## 📱 접속 URL

### 로컬 접속
```
http://localhost:5001
```

### 네트워크 접속 (다른 기기)
```
http://[Mac-IP-주소]:5001
```

**Mac IP 확인:**
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

---

## ✅ 정상 실행 확인

서버가 정상적으로 실행되면 다음과 같은 로그가 표시됩니다:

```
2025-11-10 19:06:56,495 - INFO - 🌐 웹 기반 감정 분석 어플리케이션 시작
2025-11-10 19:06:56,496 - INFO - 🚀 감정 분석 서비스 시작됨
 * Serving Flask app 'app'
 * Debug mode: off
 * Running on http://0.0.0.0:5001 (Press CTRL+C to quit)
```

**브라우저에서 http://localhost:5001 접속!**

---

## 🛑 서버 종료

```bash
Ctrl + C
```

---

## 💡 팁

### 백그라운드 실행
```bash
nohup python3 app.py > app.log 2>&1 &
```

### 로그 확인
```bash
tail -f app.log
```

### 백그라운드 프로세스 종료
```bash
ps aux | grep app.py
kill <PID>
```

---

**모든 준비 완료! 지금 실행하세요:** 🎉

```bash
python3 app.py
```

