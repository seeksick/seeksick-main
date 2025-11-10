# 🔐 환경 설정 가이드

## 📌 개요

이 프로젝트는 OpenAI ChatGPT API를 사용하여 공감적인 대화를 제공합니다.  
API 키는 **환경변수**로 관리되어 GitHub에 노출되지 않습니다.

---

## ⚙️ 설정 방법

### 1. `.env` 파일 생성

프로젝트 루트 디렉토리에 `.env` 파일을 생성하세요:

```bash
cp env.example .env
```

### 2. API 키 입력

`.env` 파일을 열고 OpenAI API 키를 입력하세요:

```bash
# OpenAI API Key
OPENAI_API_KEY=your_actual_api_key_here
```

### 3. API 키 발급 방법

1. [OpenAI Platform](https://platform.openai.com/api-keys)에 접속
2. 로그인 후 "Create new secret key" 클릭
3. 생성된 키를 복사하여 `.env` 파일에 붙여넣기

---

## 📦 필수 패키지 설치

환경변수를 읽기 위해 `python-dotenv`가 필요합니다:

```bash
pip install -r requirements-web.txt
```

또는 직접 설치:

```bash
pip install python-dotenv
```

---

## 🚀 실행

환경변수 설정 후 앱을 실행하세요:

```bash
python3 app.py
```

브라우저에서 접속:
```
http://localhost:5001
```

---

## ⚠️ 중요 사항

### ✅ 해야 할 것
- `.env` 파일에 API 키 저장
- `.env` 파일은 **절대 GitHub에 커밋하지 않기**
- `env.example` 파일은 커밋 (템플릿용)

### ❌ 하지 말아야 할 것
- API 키를 코드에 직접 작성
- `.env` 파일을 GitHub에 푸시
- API 키를 공개 저장소에 노출

---

## 🔍 `.gitignore` 확인

`.env` 파일이 `.gitignore`에 포함되어 있는지 확인하세요:

```bash
# .gitignore
.env
config.py
```

---

## 🛠️ 문제 해결

### API 키가 없을 때

```
⚠️ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.
ChatGPT 기능이 비활성화됩니다.
```

**해결 방법:**
1. `.env` 파일이 프로젝트 루트에 있는지 확인
2. `OPENAI_API_KEY=` 뒤에 실제 키가 입력되어 있는지 확인
3. `python-dotenv` 패키지가 설치되어 있는지 확인

### 환경변수가 로드되지 않을 때

```bash
# Python 인터프리터에서 테스트
python3
>>> import os
>>> from dotenv import load_dotenv
>>> load_dotenv()
>>> print(os.getenv('OPENAI_API_KEY'))
```

---

## 📁 파일 구조

```
seeksick-main/
├── .env                 # API 키 (Git 제외) ⚠️
├── env.example          # 템플릿 (Git 포함) ✅
├── .gitignore           # .env 제외 설정 ✅
├── app.py               # 환경변수 사용
└── requirements-web.txt # python-dotenv 포함
```

---

## 💡 보안 팁

1. **API 키 회전**: 주기적으로 API 키를 재발급하세요
2. **사용량 모니터링**: OpenAI 대시보드에서 사용량 확인
3. **비용 제한**: OpenAI 설정에서 월 사용 한도 설정
4. **Git 이력 확인**: 실수로 커밋한 키가 있는지 확인

```bash
# Git 이력에서 API 키 검색
git log -p | grep -i "api_key"
```

키가 노출되었다면:
1. 즉시 OpenAI에서 해당 키 삭제
2. 새 키 발급
3. Git 이력 정리 (필요 시)

---

## ✅ 완료!

이제 API 키가 안전하게 관리되며 GitHub에 노출되지 않습니다! 🎉

