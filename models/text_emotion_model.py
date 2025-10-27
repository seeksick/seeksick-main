#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
텍스트 감정 분석 모델 (seeksick-kobert.pt)

KoBERT 기반의 한국어 텍스트 감정 분류 모델
감정: [행복, 우울, 놀람, 화남, 중립]
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class KoBERTForEmotion(nn.Module):
    """KoBERT 기반 감정 분류 모델"""
    
    def __init__(self, model_name: str = "monologg/kobert", 
                 num_emotions: int = 5, dropout_rate: float = 0.3):
        super(KoBERTForEmotion, self).__init__()
        
        # KoBERT 모델
        self.bert = BertModel.from_pretrained(model_name)
        
        # 분류기
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_emotions)
        
        # 가중치 초기화
        self._init_weights()
        
    def _init_weights(self):
        """가중치 초기화"""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # BERT 인코딩
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # [CLS] 토큰의 hidden state 사용
        cls_output = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]
        
        # 드롭아웃 적용
        cls_output = self.dropout(cls_output)
        
        # 분류
        logits = self.classifier(cls_output)
        
        return logits

class TextPreprocessor:
    """텍스트 전처리기"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def clean_text(self, text: str) -> str:
        """텍스트 정제"""
        import re
        
        # 기본 정제
        text = text.strip()
        
        # 중복 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 특수문자 정규화 (선택적)
        # text = re.sub(r'[^\w\s가-힣]', '', text)
        
        return text
    
    def tokenize_text(self, text: str, max_length: int = 128) -> Dict[str, torch.Tensor]:
        """텍스트 토크나이징"""
        cleaned_text = self.clean_text(text)
        
        encoded = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        )
        
        return encoded

class TextEmotionAnalyzer:
    """텍스트 감정 분석기"""
    
    def __init__(self, model_path: str = "models/seeksick-kobert.pt"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.preprocessor = None
        self.emotions = ['happy', 'depressed', 'surprised', 'angry', 'neutral']
        
        self._load_tokenizer()
        self._load_model()
        
    def _load_tokenizer(self):
        """토크나이저 로드"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "monologg/kobert", 
                trust_remote_code=True
            )
            self.preprocessor = TextPreprocessor(self.tokenizer)
            logger.info("KoBERT 토크나이저를 로드했습니다.")
            
        except Exception as e:
            logger.error(f"토크나이저 로드 실패: {e}")
            self.tokenizer = None
    
    def _load_model(self):
        """텍스트 감정 분석 모델 로드"""
        if self.tokenizer is None:
            logger.error("토크나이저가 로드되지 않아 모델을 로드할 수 없습니다.")
            return
            
        try:
            self.model = KoBERTForEmotion(num_emotions=5)
            
            if os.path.exists(self.model_path):
                # 모델 가중치 로드
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        # state_dict 자체인 경우
                        state_dict = checkpoint
                else:
                    # 전체 모델이 저장된 경우
                    self.model = checkpoint
                    self.model.to(self.device)
                    self.model.eval()
                    logger.info(f"텍스트 감정 모델을 로드했습니다: {self.model_path}")
                    return
                
                # 키 이름 변환: kobert -> bert
                converted_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('kobert.'):
                        new_key = key.replace('kobert.', 'bert.', 1)
                        converted_state_dict[new_key] = value
                    else:
                        converted_state_dict[key] = value
                
                # 모델에 로드
                self.model.load_state_dict(converted_state_dict)
                logger.info(f"텍스트 감정 모델을 로드했습니다: {self.model_path}")
            else:
                logger.warning(f"텍스트 감정 모델 파일을 찾을 수 없습니다: {self.model_path}")
                logger.info("사전 훈련된 KoBERT 모델을 사용합니다.")
                
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"텍스트 감정 모델 로드 실패: {e}")
            self.model = None
    
    def analyze_text(self, text: str) -> Optional[np.ndarray]:
        """텍스트 감정 분석"""
        if self.model is None or not text.strip():
            return None
            
        try:
            # 텍스트 전처리 및 토크나이징
            inputs = self.preprocessor.tokenize_text(text)
            
            # 디바이스로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 추론
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
                
            return probs
            
        except Exception as e:
            logger.error(f"텍스트 감정 분석 실패: {e}")
            return None
    
    def analyze_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """배치 텍스트 감정 분석"""
        if self.model is None:
            return [None] * len(texts)
            
        results = []
        
        try:
            # 배치 토크나이징
            cleaned_texts = [self.preprocessor.clean_text(text) for text in texts]
            
            encoded = self.tokenizer(
                cleaned_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
                add_special_tokens=True
            )
            
            # 디바이스로 이동
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # 배치 추론
            with torch.no_grad():
                outputs = self.model(**encoded)
                probs_batch = F.softmax(outputs, dim=1).cpu().numpy()
                
            results = [probs for probs in probs_batch]
            
        except Exception as e:
            logger.error(f"배치 텍스트 감정 분석 실패: {e}")
            results = [None] * len(texts)
            
        return results
    
    def get_emotion_label(self, probs: np.ndarray) -> tuple:
        """확률에서 감정 라벨 추출"""
        max_idx = np.argmax(probs)
        emotion = self.emotions[max_idx]
        confidence = probs[max_idx]
        return emotion, confidence
    
    def get_top_emotions(self, probs: np.ndarray, top_k: int = 3) -> List[tuple]:
        """상위 K개 감정 반환"""
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_emotions = []
        
        for idx in top_indices:
            emotion = self.emotions[idx]
            confidence = probs[idx]
            top_emotions.append((emotion, confidence))
            
        return top_emotions
    
    def analyze_with_confidence_threshold(self, text: str, 
                                        threshold: float = 0.5) -> Optional[tuple]:
        """신뢰도 임계값을 사용한 감정 분석"""
        probs = self.analyze_text(text)
        
        if probs is None:
            return None
            
        emotion, confidence = self.get_emotion_label(probs)
        
        if confidence >= threshold:
            return emotion, confidence, probs
        else:
            return "uncertain", confidence, probs

class EmotionKeywordExtractor:
    """감정 키워드 추출기"""
    
    def __init__(self):
        # 감정별 키워드 사전
        self.emotion_keywords = {
            'happy': [
                '기쁘다', '좋다', '행복', '웃음', '즐겁다', '신나다', '만족',
                '환상', '완벽', '멋지다', '훌륭', '최고', '사랑', '감사',
                'ㅎㅎ', 'ㅋㅋ', '하하', '굿', '좋아', '최고야'
            ],
            'depressed': [
                '슬프다', '우울', '힘들다', '괴롭다', '절망', '상처', '눈물',
                '외롭다', '공허', '무기력', '지치다', '포기', '실망', '좌절',
                'ㅠㅠ', 'ㅜㅜ', '흑흑', '아..', '하..', '싫어'
            ],
            'surprised': [
                '놀랍다', '깜짝', '어?', '헉', '와', '오', '진짜?', '정말?',
                '어머', '세상에', '믿을 수 없다', '신기', '대박', '어떻게',
                '!', '!!', '???', '어쩌지', '갑자기'
            ],
            'angry': [
                '화나다', '짜증', '분노', '열받다', '싫다', '바보', '미치다',
                '답답', '빡치다', '억울', '욕', '죽이다', '때리다', '없애다',
                '제발', '진짜', '왜', '어떻게', '안된다', '못하다'
            ],
            'neutral': [
                '그냥', '보통', '평범', '일반', '그런데', '그러면', '아마',
                '생각', '말하다', '이야기', '설명', '내용', '정보', '방법',
                '시간', '장소', '사람', '일', '것', '수', '때'
            ]
        }
    
    def extract_emotion_keywords(self, text: str) -> Dict[str, List[str]]:
        """텍스트에서 감정 키워드 추출"""
        found_keywords = {emotion: [] for emotion in self.emotions}
        
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    found_keywords[emotion].append(keyword)
        
        return found_keywords
    
    def get_keyword_based_emotion(self, text: str) -> Optional[str]:
        """키워드 기반 감정 예측"""
        keyword_counts = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text)
            keyword_counts[emotion] = count
        
        if max(keyword_counts.values()) > 0:
            return max(keyword_counts, key=keyword_counts.get)
        else:
            return None

def test_text_emotion_analyzer():
    """텍스트 감정 분석기 테스트"""
    analyzer = TextEmotionAnalyzer()
    
    # 테스트 문장들
    test_sentences = [
        ("happy", "오늘 정말 기분이 좋아! 모든 일이 잘 풀리고 있어."),
        ("depressed", "너무 힘들고 우울해... 아무것도 하기 싫어."),
        ("surprised", "어? 진짜? 정말 깜짝 놀랐어! 믿을 수 없다."),
        ("angry", "정말 화가 나! 왜 이런 일이 생기는 거야?"),
        ("neutral", "오늘 날씨가 흐리네요. 비가 올 것 같아요."),
    ]
    
    print("텍스트 감정 분석 테스트를 시작합니다.")
    print("=" * 60)
    
    for true_emotion, sentence in test_sentences:
        print(f"\n문장: {sentence}")
        print(f"실제 감정: {true_emotion}")
        
        # 감정 분석
        probs = analyzer.analyze_text(sentence)
        
        if probs is not None:
            pred_emotion, confidence = analyzer.get_emotion_label(probs)
            
            print(f"예측 감정: {pred_emotion} (신뢰도: {confidence:.3f})")
            print("감정별 확률:")
            for emotion, prob in zip(analyzer.emotions, probs):
                print(f"  {emotion}: {prob:.3f}")
                
            # 상위 3개 감정
            top_emotions = analyzer.get_top_emotions(probs, top_k=3)
            print("상위 3개 감정:")
            for i, (emotion, conf) in enumerate(top_emotions, 1):
                print(f"  {i}. {emotion}: {conf:.3f}")
        else:
            print("감정 분석에 실패했습니다.")
        
        print("-" * 40)
    
    # 사용자 입력 테스트
    print("\n사용자 입력 테스트 (종료: 'quit')")
    while True:
        user_input = input("\n문장을 입력하세요: ")
        
        if user_input.lower() in ['quit', 'exit', '종료']:
            break
            
        probs = analyzer.analyze_text(user_input)
        
        if probs is not None:
            emotion, confidence = analyzer.get_emotion_label(probs)
            print(f"예측 감정: {emotion} (신뢰도: {confidence:.3f})")
            
            print("감정별 확률:")
            for emo, prob in zip(analyzer.emotions, probs):
                print(f"  {emo}: {prob:.3f}")
        else:
            print("감정 분석에 실패했습니다.")
    
    print("텍스트 감정 분석 테스트를 종료합니다.")

if __name__ == "__main__":
    test_text_emotion_analyzer()
