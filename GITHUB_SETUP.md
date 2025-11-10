# 🚀 GitHub에 업로드하기

## ✅ 준비 완료!

API 키가 안전하게 환경변수로 관리되도록 설정이 완료되었습니다.

---

## 📋 변경 사항

### 1. **환경변수 관리**
```python
# app.py
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)
```

### 2. **.gitignore 업데이트**
```
# 환경변수 (API 키 등)
.env
config.py

# 백업 파일
*_backup.py
*_old.py
app_text_only.py
app_optimized.py
```

### 3. **파일 구조**
```
seeksick-main/
├── .env                 # API 키 저장 (Git 제외) ⚠️
├── env.example          # 템플릿 (Git 포함) ✅
├── .gitignore           # .env 제외 설정
├── app.py               # 환경변수 사용
├── requirements-web.txt # python-dotenv 추가
└── SETUP_ENV.md         # 환경 설정 가이드
```

---

## 🔐 GitHub 업로드 전 체크리스트

### ✅ 확인 사항

1. **API 키 제거**
   ```bash
   # app.py에 하드코딩된 API 키가 없는지 확인
   grep -r "sk-proj" .
   grep -r "api_key=" app.py
   ```

2. **.env 파일 확인**
   ```bash
   # .env 파일이 .gitignore에 포함되어 있는지 확인
   cat .gitignore | grep ".env"
   ```

3. **Git 상태 확인**
   ```bash
   git status
   # .env 파일이 "Untracked files"에 없어야 함
   ```

---

## 📤 GitHub 업로드 단계

### 1. Git 초기화 (처음이라면)
```bash
cd /Users/qkrwnsmir/Desktop/seeksick-main
git init
```

### 2. 변경사항 스테이징
```bash
# 모든 파일 추가 (.gitignore에 의해 .env는 자동 제외)
git add .

# 상태 확인
git status
```

### 3. 커밋
```bash
git commit -m "feat: 실시간 멀티모달 감정 분석 웹 어플리케이션

- 얼굴 표정 실시간 분석 (ResNet18, 74% 정확도)
- 음성 입력 시 3개 모달리티 통합 (얼굴+음성+텍스트)
- Late Fusion 가중 평균 (2초 주기)
- ChatGPT 공감적 대화 (API 키 환경변수 관리)
- Web Speech API 실시간 음성 인식
"
```

### 4. GitHub 저장소 생성
1. [GitHub](https://github.com) 접속
2. "New repository" 클릭
3. 저장소 이름 입력 (예: `seeksick-emotion-analysis`)
4. Public/Private 선택
5. "Create repository" 클릭

### 5. 원격 저장소 연결 및 푸시
```bash
# 원격 저장소 추가
git remote add origin https://github.com/your-username/seeksick-emotion-analysis.git

# 메인 브랜치로 변경 (필요 시)
git branch -M main

# 푸시
git push -u origin main
```

---

## 📝 README.md 작성

GitHub 저장소에 다음 내용을 포함하세요:

```markdown
# 🎭 실시간 멀티모달 감정 분석 시스템

## 🌟 주요 기능

### 1. 실시간 얼굴 표정 분석 📹
- ResNet18 (74% 정확도)
- 웹캠 10 FPS
- 2초마다 화면 업데이트

### 2. 음성 기반 통합 분석 🎤
- 음성 입력 시 3개 모달리티 사용
  - 얼굴 표정 (실시간 최신값)
  - 음성 톤 (Wav2Vec2, 65%)
  - 텍스트 내용 (KoBERT, 66%)

### 3. Late Fusion 가중 평균
- 모델 정확도 기반 가중치 설정
- 2초 주기로 감정 통합

### 4. ChatGPT 공감적 대화 💬
- 실시간 감정 분석 결과 반영
- 멀티모달 컨텍스트 전달

## 🚀 설치 및 실행

### 1. 환경 설정
\`\`\`bash
# 의존성 설치
pip install -r requirements.txt
pip install -r requirements-web.txt

# API 키 설정
cp env.example .env
# .env 파일에 OpenAI API 키 입력
\`\`\`

### 2. 실행
\`\`\`bash
python3 app.py
# http://localhost:5001
\`\`\`

## 📚 상세 가이드
- [환경 설정](SETUP_ENV.md)
- [웹 어플리케이션 가이드](WEB_APP_GUIDE.md)
- [Late Fusion 설명](WEIGHTED_FUSION_GUIDE.md)

## 🔒 보안
- API 키는 환경변수로 관리
- `.env` 파일은 Git에 포함되지 않음
```

---

## ⚠️ 주의사항

### 실수로 API 키를 커밋한 경우

1. **즉시 키 폐기**
   - [OpenAI API Keys](https://platform.openai.com/api-keys)
   - 해당 키 삭제

2. **새 키 발급**
   - 새 키 생성
   - `.env` 파일 업데이트

3. **Git 이력 정리 (필요 시)**
   ```bash
   # BFG Repo-Cleaner 사용
   # 또는 Git filter-branch
   # (복잡하므로 가능하면 새 저장소 생성 권장)
   ```

---

## ✅ 완료!

이제 GitHub에 안전하게 업로드할 수 있습니다! 🎉

### 다른 사용자가 클론한 후

1. 저장소 클론
   ```bash
   git clone https://github.com/your-username/seeksick-emotion-analysis.git
   cd seeksick-emotion-analysis
   ```

2. 환경 설정
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-web.txt
   cp env.example .env
   # .env 파일에 자신의 API 키 입력
   ```

3. 실행
   ```bash
   python3 app.py
   ```

