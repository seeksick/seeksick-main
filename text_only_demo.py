#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
텍스트 감정 분석 전용 - 웹캠/마이크 없이 사용 가능
터미널에서 바로 실행 가능한 버전
"""

from models.text_emotion_model import TextEmotionAnalyzer

def analyze_text(text, analyzer):
    """텍스트 감정 분석 및 결과 출력"""
    probs = analyzer.analyze_text(text)
    
    if probs is not None:
        emotion, confidence = analyzer.get_emotion_label(probs)
        
        # 이모지 매핑
        emoji_map = {
            'happy': '😊',
            'depressed': '😢',
            'surprised': '😮',
            'angry': '😠',
            'neutral': '😐'
        }
        
        emoji = emoji_map.get(emotion, '❓')
        
        print(f"\n   {emoji} 예측 감정: {emotion.upper()}")
        print(f"   📊 신뢰도: {confidence:.1%}")
        
        # 상위 3개
        top_3 = analyzer.get_top_emotions(probs, top_k=3)
        print(f"\n   🏆 상위 3개 감정:")
        for i, (emo, conf) in enumerate(top_3, 1):
            e = emoji_map.get(emo, '❓')
            print(f"      {i}위: {e} {emo:12s} {conf:.1%}")
        
        print(f"\n   {'─'*50}")
        return emotion, confidence
    else:
        print("   ❌ 분석 실패")
        return None, None

def main():
    """메인 함수"""
    print("="*70)
    print("📝 텍스트 감정 분석 - 터미널 전용 버전")
    print("="*70)
    print()
    print("웹캠이나 마이크 없이도 사용할 수 있습니다!")
    print()
    
    # 모델 로드
    print("📥 KoBERT 모델 로딩 중...")
    analyzer = TextEmotionAnalyzer()
    
    if analyzer.model is None:
        print("❌ 모델 로드 실패!")
        return
    
    print("✅ 모델 로드 완료!\n")
    
    # 데모 문장들
    print("="*70)
    print("🧪 데모 문장 자동 분석")
    print("="*70)
    
    demo_texts = [
        ("😊 긍정적인 문장", "오늘 정말 기분이 좋아요! 모든 일이 잘 풀려요!"),
        ("😢 슬픈 문장", "너무 힘들고 우울해서 눈물이 나요..."),
        ("😮 놀란 문장", "와! 대박! 진짜 믿을 수가 없어요!"),
        ("😠 화난 문장", "정말 화가 나서 참을 수가 없어요!"),
        ("😐 평범한 문장", "오늘 날씨가 흐리네요."),
        ("💼 비즈니스", "회의 일정을 확인해 주세요."),
        ("🎉 축하 문장", "생일 축하해요! 정말 멋진 날이에요!"),
        ("😰 걱정 문장", "걱정되고 불안해서 잠을 잘 수가 없어요."),
    ]
    
    results = []
    
    for title, text in demo_texts:
        print(f"\n{title}")
        print(f"💬 문장: '{text}'")
        emotion, conf = analyze_text(text, analyzer)
        if emotion:
            results.append((text, emotion, conf))
    
    # 요약
    print("\n" + "="*70)
    print("📊 분석 결과 요약")
    print("="*70)
    print()
    
    emotion_counts = {}
    for _, emotion, _ in results:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    print("감정별 분포:")
    emoji_map = {
        'happy': '😊',
        'depressed': '😢',
        'surprised': '😮',
        'angry': '😠',
        'neutral': '😐'
    }
    
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        emoji = emoji_map.get(emotion, '❓')
        bar = '█' * count
        print(f"  {emoji} {emotion:12s} {bar} ({count}개)")
    
    print()
    print("="*70)
    print("✅ 분석 완료!")
    print()
    print("💡 사용자 입력 모드는 'python test_realtime.py' 를 실행하세요.")
    print("="*70)

if __name__ == "__main__":
    main()

