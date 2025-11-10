#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
실시간 멀티모달 감정 분석 어플리케이션
- 얼굴: 실시간 웹캠 분석 (화면에 계속 반영)
- 음성: 음성 입력 시에만 분석
- GPT: 음성 입력 시 3개 모달리티(얼굴+음성+텍스트) 모두 사용
"""

from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS
import threading
import queue
import time
import numpy as np
import json
import logging
from datetime import datetime
from openai import OpenAI
import cv2
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 기존 모델 임포트
from models import EMOTIONS
from models.face_emotion_model import FaceEmotionAnalyzer
from models.voice_emotion_model import VoiceEmotionAnalyzer
from models.text_emotion_model import TextEmotionAnalyzer
from main import LateFusion

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 전역 상수
FUSION_INTERVAL = 2.0  # Late Fusion 간격 (2초)
CHAT_PERSONALITY = """당신은 공감적이고 따뜻한 AI 상담사입니다. 
사용자의 감정을 이해하고 위로와 격려를 제공합니다.

**중요**: 사용자의 현재 감정은 Late Fusion 기술로 분석되었습니다.
- 얼굴 표정 (ResNet18, 74% 정확도) - 실시간 웹캠 분석
- 음성 톤 (Wav2Vec2, 65% 정확도) - 음성 입력 시 분석
- 텍스트 내용 (KoBERT, 66% 정확도) - 음성→텍스트 변환

이 세 가지 모달리티를 가중 평균하여 실시간으로 분석된 감정 결과를 바탕으로 
사용자의 진짜 감정 상태를 파악하고 적절하게 반응하세요.

감정 분포를 참고하여:
- 주요 감정에 공감하되, 다른 감정들도 고려하세요
- 감정이 혼재된 경우 복합적으로 이해하세요
- 실시간 얼굴 표정 변화를 고려하여 자연스럽게 대화하세요"""

# OpenAI 클라이언트 초기화
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logger.warning("⚠️ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    logger.info("ChatGPT 기능이 비활성화됩니다.")
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

# 감정 데이터 큐 (SSE용)
emotion_queue = queue.Queue(maxsize=100)

# 채팅 히스토리
chat_history = []


class RealtimeEmotionService:
    """실시간 감정 분석 서비스"""
    
    def __init__(self):
        self.is_running = False
        self.face_thread = None
        self.fusion_thread = None
        
        # Late Fusion 초기화
        self.late_fusion = LateFusion(interval=FUSION_INTERVAL)
        
        # 최신 감정 데이터
        self.latest_emotions = {
            "happy": 0.2,
            "depressed": 0.2,
            "surprised": 0.2,
            "angry": 0.2,
            "neutral": 0.2
        }
        
        # 최신 모달리티별 감정 (음성 입력 시 사용)
        self.latest_face_emotion = None
        self.latest_voice_emotion = None
        self.latest_text_emotion = None
        
        # 모델 로드
        try:
            logger.info("감정 분석 모델 로드 중...")
            self.face_analyzer = FaceEmotionAnalyzer()
            self.voice_analyzer = VoiceEmotionAnalyzer()
            self.text_analyzer = TextEmotionAnalyzer()
            logger.info("✅ 모든 모델 로드 완료")
            self.models_loaded = True
        except Exception as e:
            logger.warning(f"⚠️ 모델 로드 실패: {e}")
            self.models_loaded = False
    
    def start(self):
        """감정 분석 서비스 시작"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 얼굴 분석 스레드 (실시간)
        self.face_thread = threading.Thread(target=self._face_analysis_loop, daemon=True)
        self.face_thread.start()
        
        # Fusion 스레드
        self.fusion_thread = threading.Thread(target=self._fusion_loop, daemon=True)
        self.fusion_thread.start()
        
        logger.info("🚀 실시간 감정 분석 서비스 시작")
    
    def stop(self):
        """감정 분석 서비스 중지"""
        self.is_running = False
        if self.face_thread:
            self.face_thread.join(timeout=2)
        if self.fusion_thread:
            self.fusion_thread.join(timeout=2)
        logger.info("🛑 감정 분석 서비스 중지")
    
    def _face_analysis_loop(self):
        """실시간 얼굴 분석 루프 (웹캠)"""
        if not self.models_loaded:
            logger.warning("모델 미로드 - 얼굴 분석 스킵")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("웹캠을 열 수 없습니다")
            return
        
        logger.info("📹 실시간 얼굴 분석 시작")
        
        while self.is_running:
            try:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                # 얼굴 감정 분석
                result = self.face_analyzer.analyze_face(frame)
                
                if result is None:
                    time.sleep(0.1)
                    continue
                
                face_probs, _ = result
                
                if face_probs is not None:
                    # Late Fusion 버퍼에 추가
                    self.late_fusion.add_face_emotion(face_probs)
                    
                    # 최신 얼굴 감정 저장
                    self.latest_face_emotion = {
                        EMOTIONS[i]: float(face_probs[i])
                        for i in range(len(EMOTIONS))
                    }
                
                time.sleep(0.1)  # 10 FPS
                
            except Exception as e:
                logger.error(f"얼굴 분석 에러: {e}")
                time.sleep(1)
        
        cap.release()
        logger.info("📹 얼굴 분석 종료")
    
    def _fusion_loop(self):
        """Late Fusion 루프 (2초마다)"""
        while self.is_running:
            try:
                time.sleep(0.5)
                
                # Late Fusion 수행
                if self.late_fusion.should_fuse():
                    fusion_result = self.late_fusion.fuse_emotions()
                    if fusion_result is not None:
                        emotion, all_probs, modalities = fusion_result
                        
                        # 감정 데이터 업데이트
                        self.latest_emotions = {
                            EMOTIONS[i]: float(all_probs[i])
                            for i in range(len(EMOTIONS))
                        }
                        
                        # SSE 큐에 푸시
                        emotion_data = {
                            "timestamp": datetime.now().isoformat(),
                            "emotions": self.latest_emotions,
                            "primary_emotion": emotion,
                            "modalities": modalities
                        }
                        
                        try:
                            emotion_queue.put_nowait(emotion_data)
                        except queue.Full:
                            emotion_queue.get()
                            emotion_queue.put_nowait(emotion_data)
                        
                        logger.info(f"📊 감정 업데이트: {emotion} ({self.latest_emotions[emotion]:.1%}) [모달리티: {modalities}]")
                    
                    self.late_fusion.reset_buffers()
                
            except Exception as e:
                logger.error(f"Fusion 루프 에러: {e}")
                time.sleep(1)
    
    def process_voice_input(self, text: str, audio_data=None):
        """
        음성 입력 처리 (3개 모달리티 모두 사용)
        
        Args:
            text: 음성→텍스트 변환 결과
            audio_data: 음성 오디오 데이터 (옵션)
        
        Returns:
            dict: 3개 모달리티 감정 결과
        """
        if not text or not text.strip():
            return None
        
        result = {
            "face": self.latest_face_emotion,
            "voice": None,
            "text": None
        }
        
        try:
            # 1. 텍스트 감정 분석
            if self.models_loaded:
                text_probs = self.text_analyzer.analyze_text(text)
                if text_probs is not None:
                    self.late_fusion.add_text_emotion(text_probs)
                    result["text"] = {
                        EMOTIONS[i]: float(text_probs[i])
                        for i in range(len(EMOTIONS))
                    }
                    self.latest_text_emotion = result["text"]
            
            # 2. 음성 감정 분석 (audio_data가 있으면)
            if audio_data is not None and self.models_loaded:
                voice_probs = self.voice_analyzer.analyze_audio(audio_data)
                if voice_probs is not None:
                    self.late_fusion.add_voice_emotion(voice_probs)
                    result["voice"] = {
                        EMOTIONS[i]: float(voice_probs[i])
                        for i in range(len(EMOTIONS))
                    }
                    self.latest_voice_emotion = result["voice"]
            
            # 로그 출력
            logger.info(f"🎤 [음성 입력] {text}")
            if result["face"]:
                face_top = max(result["face"], key=result["face"].get)
                logger.info(f"   ├─ 얼굴: {face_top} ({result['face'][face_top]:.1%})")
            if result["voice"]:
                voice_top = max(result["voice"], key=result["voice"].get)
                logger.info(f"   ├─ 음성: {voice_top} ({result['voice'][voice_top]:.1%})")
            if result["text"]:
                text_top = max(result["text"], key=result["text"].get)
                logger.info(f"   └─ 텍스트: {text_top} ({result['text'][text_top]:.1%})")
            
            return result
            
        except Exception as e:
            logger.error(f"음성 입력 처리 에러: {e}")
            return result
    
    def process_text_input(self, text: str):
        """
        텍스트 입력 처리 (텍스트만 사용)
        
        Args:
            text: 입력 텍스트
        
        Returns:
            dict: 텍스트 감정 결과
        """
        if not text or not text.strip():
            return None
        
        try:
            if self.models_loaded:
                text_probs = self.text_analyzer.analyze_text(text)
                if text_probs is not None:
                    self.late_fusion.add_text_emotion(text_probs)
                    
                    result = {
                        EMOTIONS[i]: float(text_probs[i])
                        for i in range(len(EMOTIONS))
                    }
                    
                    text_top = max(result, key=result.get)
                    logger.info(f"⌨️  [텍스트 입력] {text}")
                    logger.info(f"   └─ 텍스트: {text_top} ({result[text_top]:.1%})")
                    
                    return result
            
            return self.latest_emotions
            
        except Exception as e:
            logger.error(f"텍스트 처리 에러: {e}")
            return self.latest_emotions
    
    def get_latest_emotions(self):
        """최신 감정 데이터 반환"""
        return self.latest_emotions
    
    def get_multimodal_emotions(self):
        """모달리티별 최신 감정 반환"""
        return {
            "face": self.latest_face_emotion,
            "voice": self.latest_voice_emotion,
            "text": self.latest_text_emotion
        }


# 전역 서비스 인스턴스
emotion_service = RealtimeEmotionService()


def get_chatgpt_response(user_message: str, fusion_emotions: dict, modality_emotions: dict = None) -> str:
    """ChatGPT API로 공감적 응답 생성"""
    # API 키가 없으면 기본 응답 반환
    if client is None:
        return "ChatGPT 기능을 사용하려면 .env 파일에 OPENAI_API_KEY를 설정하세요."
    
    try:
        # Late Fusion 결과 파싱
        primary_emotion = max(fusion_emotions, key=fusion_emotions.get)
        emotion_confidence = fusion_emotions[primary_emotion]
        
        # 모든 감정 확률 정렬
        sorted_emotions = sorted(fusion_emotions.items(), key=lambda x: x[1], reverse=True)
        
        # 감정 컨텍스트 생성
        emotion_context = f"\n[Late Fusion 감정 분석 결과]\n"
        emotion_context += f"주요 감정: {primary_emotion} ({emotion_confidence:.1%})\n"
        emotion_context += "전체 감정 분포:\n"
        for emotion, prob in sorted_emotions:
            emotion_context += f"  - {emotion}: {prob:.1%}\n"
        
        # 모달리티별 상세 정보 (음성 입력 시)
        if modality_emotions:
            emotion_context += "\n[모달리티별 분석]\n"
            if modality_emotions.get("face"):
                face_top = max(modality_emotions["face"], key=modality_emotions["face"].get)
                emotion_context += f"얼굴 표정: {face_top} ({modality_emotions['face'][face_top]:.1%})\n"
            if modality_emotions.get("voice"):
                voice_top = max(modality_emotions["voice"], key=modality_emotions["voice"].get)
                emotion_context += f"음성 톤: {voice_top} ({modality_emotions['voice'][voice_top]:.1%})\n"
            if modality_emotions.get("text"):
                text_top = max(modality_emotions["text"], key=modality_emotions["text"].get)
                emotion_context += f"텍스트 내용: {text_top} ({modality_emotions['text'][text_top]:.1%})\n"
        
        # 대화 히스토리 준비
        messages = [
            {"role": "system", "content": CHAT_PERSONALITY + emotion_context}
        ]
        
        # 최근 3개 대화만 포함
        for msg in chat_history[-6:]:
            messages.append(msg)
        
        # 현재 메시지
        messages.append({"role": "user", "content": user_message})
        
        # ChatGPT API 호출
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=300
        )
        
        assistant_message = response.choices[0].message.content
        
        # 히스토리에 추가
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
        
    except Exception as e:
        logger.error(f"ChatGPT API 에러: {e}")
        return "죄송합니다. 잠시 후 다시 시도해주세요. 💙"


# =============================================================================
# Flask 라우트
# =============================================================================

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')


@app.route('/api/emotions')
def get_emotions():
    """현재 감정 데이터 반환"""
    emotions = emotion_service.get_latest_emotions()
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "emotions": emotions,
        "primary_emotion": max(emotions, key=emotions.get)
    })


@app.route('/api/emotions/stream')
def emotion_stream():
    """실시간 감정 데이터 스트리밍 (SSE)"""
    def generate():
        logger.info("SSE 클라이언트 연결")
        
        # 초기 데이터
        initial_data = {
            "timestamp": datetime.now().isoformat(),
            "emotions": emotion_service.get_latest_emotions(),
            "primary_emotion": max(emotion_service.get_latest_emotions(), 
                                  key=emotion_service.get_latest_emotions().get)
        }
        yield f"data: {json.dumps(initial_data)}\n\n"
        
        # 지속적 업데이트
        while True:
            try:
                data = emotion_queue.get(timeout=5)
                yield f"data: {json.dumps(data)}\n\n"
            except queue.Empty:
                yield f": keepalive\n\n"
            except GeneratorExit:
                logger.info("SSE 클라이언트 연결 종료")
                break
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    통합 채팅 API
    - 음성: 3개 모달리티(얼굴+음성+텍스트) 모두 사용
    - 텍스트: 텍스트만 사용
    """
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        is_voice = data.get('is_voice', False)
        
        if not user_message:
            return jsonify({"error": "메시지가 비어있습니다."}), 400
        
        modality_emotions = None
        
        if is_voice:
            # 음성 입력: 3개 모달리티 모두 사용
            modality_emotions = emotion_service.process_voice_input(user_message)
        else:
            # 텍스트 입력: 텍스트만 사용
            emotion_service.process_text_input(user_message)
        
        # Late Fusion 결과 가져오기
        fusion_emotions = emotion_service.get_latest_emotions()
        
        # ChatGPT 응답 생성
        primary_fusion = max(fusion_emotions, key=fusion_emotions.get)
        logger.info(f"💬 [ChatGPT 입력] Late Fusion: {primary_fusion} ({fusion_emotions[primary_fusion]:.1%})")
        
        ai_response = get_chatgpt_response(user_message, fusion_emotions, modality_emotions)
        
        return jsonify({
            "message": ai_response,
            "fusion_emotions": fusion_emotions,
            "modality_emotions": modality_emotions,
            "timestamp": datetime.now().isoformat(),
            "is_voice": is_voice
        })
        
    except Exception as e:
        logger.error(f"채팅 API 에러: {e}")
        return jsonify({"error": "서버 오류"}), 500


@app.route('/api/chat/history')
def get_chat_history():
    """채팅 히스토리 반환"""
    return jsonify({
        "history": chat_history[-20:],
        "count": len(chat_history)
    })


@app.route('/api/chat/clear', methods=['POST'])
def clear_chat_history():
    """채팅 히스토리 초기화"""
    global chat_history
    chat_history = []
    return jsonify({"status": "success"})


@app.route('/api/service/status')
def service_status():
    """서비스 상태 확인"""
    return jsonify({
        "running": emotion_service.is_running,
        "models_loaded": emotion_service.models_loaded,
        "timestamp": datetime.now().isoformat()
    })


# =============================================================================
# 앱 시작/종료
# =============================================================================

def on_startup():
    """앱 시작"""
    logger.info("="*80)
    logger.info("🌐 실시간 멀티모달 감정 분석 어플리케이션 시작")
    logger.info("="*80)
    emotion_service.start()


def on_shutdown():
    """앱 종료"""
    logger.info("앱 종료 중...")
    emotion_service.stop()


if __name__ == '__main__':
    import os
    try:
        on_startup()
        port = int(os.environ.get('PORT', 5001))
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        logger.info("\n사용자 중단")
    finally:
        on_shutdown()

