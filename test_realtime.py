#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
실시간 텍스트 감정 분석 테스트
사용자가 입력한 문장의 감정을 실시간으로 분석합니다.
"""

import sys
from models.text_emotion_model import TextEmotionAnalyzer

def print_emotion_bar(emotion, prob, max_width=40):
    """감정 확률을 바 그래프로 표시"""
    filled = int(prob * max_width)
    bar = "█" * filled + "░" * (max_width - filled)
    
    # 감정별 이모지
    emojis = {
        'happy': '😊',
        'depressed': '😢',
        'surprised': '😮',
        'angry': '😠',
        'neutral': '😐'
    }
    
    emoji = emojis.get(emotion, '❓')
    return f"{emoji} {emotion:12s} [{bar}] {prob:.1%}"

def main():
    """메인 함수"""
    print("=" * 70)
    print("🎯 실시간 텍스트 감정 분석 테스트")
    print("=" * 70)
    print()
    print("📥 KoBERT 모델 로딩 중...")
    
    analyzer = TextEmotionAnalyzer()
    
    if analyzer.model is None:
        print("❌ 모델 로드 실패!")
        return
    
    print("✅ 모델 로드 완료!\n")
    print("=" * 70)
    print("💬 문장을 입력하면 감정을 분석합니다.")
    print("   종료하려면 'quit', 'exit', '종료' 를 입력하세요.")
    print("=" * 70)
    print()
    
    # 예제 문장 제시
    examples = [
        "오늘 정말 기분이 좋아요!",
        "너무 슬프고 우울해요...",
        "와! 대박! 정말 놀라워요!",
        "진짜 화가 나네요!",
        "그냥 평범한 하루네요."
    ]
    
    print("📝 예제 문장:")
    for i, example in enumerate(examples, 1):
        print(f"   {i}. {example}")
    print()
    
    analysis_count = 0
    
    while True:
        try:
            # 사용자 입력
            user_input = input("💭 문장 입력: ").strip()
            
            # 종료 조건
            if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                break
            
            # 빈 입력 무시
            if not user_input:
                continue
            
            # 숫자 입력 시 예제 문장 사용
            if user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(examples):
                    user_input = examples[idx]
                    print(f"   → 선택: {user_input}")
                else:
                    print("   ⚠️ 1-5 사이의 숫자를 입력하세요.")
                    continue
            
            print()
            print("   ⏳ 분석 중...")
            
            # 감정 분석
            probs = analyzer.analyze_text(user_input)
            
            if probs is not None:
                analysis_count += 1
                
                # 예측 결과
                emotion, confidence = analyzer.get_emotion_label(probs)
                
                print()
                print("   " + "─" * 66)
                print(f"   🎯 예측 감정: {emotion.upper()} (신뢰도: {confidence:.1%})")
                print("   " + "─" * 66)
                print()
                print("   📊 감정별 확률:")
                
                # 모든 감정 확률 표시
                for i, (emo, prob) in enumerate(zip(analyzer.emotions, probs)):
                    print(f"      {print_emotion_bar(emo, prob)}")
                
                print()
                
                # 상위 3개 감정
                top_emotions = analyzer.get_top_emotions(probs, top_k=3)
                print("   🏆 상위 3개:")
                for rank, (emo, conf) in enumerate(top_emotions, 1):
                    emoji = {'happy': '😊', 'depressed': '😢', 'surprised': '😮', 
                             'angry': '😠', 'neutral': '😐'}.get(emo, '❓')
                    print(f"      {rank}위: {emoji} {emo} ({conf:.1%})")
                
                print()
                
            else:
                print("   ❌ 분석 실패")
                print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Ctrl+C 감지됨")
            break
        except Exception as e:
            print(f"\n   ❌ 오류: {e}\n")
            continue
    
    # 종료 메시지
    print()
    print("=" * 70)
    print(f"📊 총 {analysis_count}개 문장 분석 완료!")
    print("👋 프로그램을 종료합니다.")
    print("=" * 70)

if __name__ == "__main__":
    main()

