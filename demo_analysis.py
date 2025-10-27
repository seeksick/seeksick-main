#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Seeksick 멀티모달 감정 분석 데모
모델 로드와 기본 기능을 테스트합니다.
"""

import sys
import numpy as np

def test_face_model():
    """얼굴 감정 모델 테스트"""
    print("\n" + "="*60)
    print("1️⃣  얼굴 감정 분석 모델 테스트")
    print("="*60)
    
    try:
        from models.face_emotion_model import FaceEmotionAnalyzer
        
        print("📥 모델 로딩 중...")
        analyzer = FaceEmotionAnalyzer()
        
        if analyzer.model is not None:
            print("✅ 얼굴 감정 모델 로드 성공!")
            print(f"   - 디바이스: {analyzer.device}")
            print(f"   - 감정 분류: {', '.join(analyzer.emotions)}")
            
            # 더미 이미지로 테스트
            print("\n🧪 더미 이미지로 추론 테스트...")
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = analyzer.analyze_face(dummy_image)
            
            if result is None:
                print("   ℹ️  얼굴이 검출되지 않았습니다 (더미 이미지이므로 정상)")
            else:
                probs, face_coords = result
                print(f"   ✅ 추론 성공! 확률 분포: {probs}")
            
            return True
        else:
            print("❌ 모델 로드 실패")
            return False
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_voice_model():
    """음성 감정 모델 테스트"""
    print("\n" + "="*60)
    print("2️⃣  음성 감정 분석 모델 테스트")
    print("="*60)
    
    try:
        from models.voice_emotion_model import VoiceEmotionAnalyzer
        
        print("📥 모델 로딩 중...")
        analyzer = VoiceEmotionAnalyzer()
        
        if analyzer.model is not None:
            print("✅ 음성 감정 모델 로드 성공!")
            print(f"   - 디바이스: {analyzer.device}")
            print(f"   - 감정 분류: {', '.join(analyzer.emotions)}")
            print(f"   - 샘플레이트: {analyzer.sample_rate} Hz")
            
            # 더미 오디오로 테스트
            print("\n🧪 더미 오디오로 추론 테스트...")
            dummy_audio = np.random.random(analyzer.sample_rate * 3).astype(np.float32)
            probs = analyzer.analyze_voice(dummy_audio)
            
            if probs is not None:
                print(f"   ✅ 추론 성공!")
                emotion, confidence = analyzer.get_emotion_label(probs)
                print(f"   예측 감정: {emotion} (신뢰도: {confidence:.3f})")
                print(f"   확률 분포:")
                for emo, prob in zip(analyzer.emotions, probs):
                    print(f"     - {emo}: {prob:.3f}")
            else:
                print("   ❌ 추론 실패")
            
            return True
        else:
            print("❌ 모델 로드 실패")
            return False
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_text_model():
    """텍스트 감정 모델 테스트"""
    print("\n" + "="*60)
    print("3️⃣  텍스트 감정 분석 모델 테스트")
    print("="*60)
    
    try:
        from models.text_emotion_model import TextEmotionAnalyzer
        
        print("📥 모델 로딩 중...")
        print("   (KoBERT 모델을 다운로드하므로 시간이 걸릴 수 있습니다)")
        analyzer = TextEmotionAnalyzer()
        
        if analyzer.model is not None:
            print("✅ 텍스트 감정 모델 로드 성공!")
            print(f"   - 디바이스: {analyzer.device}")
            print(f"   - 감정 분류: {', '.join(analyzer.emotions)}")
            
            # 테스트 문장들
            print("\n🧪 테스트 문장으로 추론...")
            test_sentences = [
                "오늘 정말 기분이 좋아요!",
                "너무 슬프고 우울해요...",
                "와! 정말 놀랍네요!",
                "화가 나서 참을 수가 없어요",
                "그냥 평범한 하루입니다.",
            ]
            
            for i, sentence in enumerate(test_sentences, 1):
                print(f"\n   [{i}] 문장: {sentence}")
                probs = analyzer.analyze_text(sentence)
                
                if probs is not None:
                    emotion, confidence = analyzer.get_emotion_label(probs)
                    print(f"       예측: {emotion} (신뢰도: {confidence:.3f})")
                    
                    # 상위 2개 감정만 표시
                    top_emotions = analyzer.get_top_emotions(probs, top_k=2)
                    print(f"       상위 감정: ", end="")
                    print(", ".join([f"{e}({c:.2f})" for e, c in top_emotions]))
                else:
                    print(f"       ❌ 분석 실패")
            
            return True
        else:
            print("❌ 모델 로드 실패")
            return False
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 함수"""
    print("🎯 Seeksick 멀티모달 감정 분석 시스템 - 데모")
    print("="*60)
    print("이 프로그램은 3가지 모달리티의 감정 분석을 테스트합니다.")
    print("="*60)
    
    results = {}
    
    # 1. 얼굴 감정 모델
    results['face'] = test_face_model()
    
    # 2. 음성 감정 모델
    results['voice'] = test_voice_model()
    
    # 3. 텍스트 감정 모델
    results['text'] = test_text_model()
    
    # 결과 요약
    print("\n" + "="*60)
    print("📊 테스트 결과 요약")
    print("="*60)
    
    status_icons = {True: "✅", False: "❌"}
    print(f"{status_icons[results['face']]} 얼굴 감정 분석: {'성공' if results['face'] else '실패'}")
    print(f"{status_icons[results['voice']]} 음성 감정 분석: {'성공' if results['voice'] else '실패'}")
    print(f"{status_icons[results['text']]} 텍스트 감정 분석: {'성공' if results['text'] else '실패'}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n총 {success_count}/{total_count} 모델 테스트 성공")
    
    if success_count == total_count:
        print("\n🎉 모든 모델이 정상적으로 작동합니다!")
        print("\n다음 단계:")
        print("  • python main.py - 실시간 멀티모달 감정 분석 실행")
        print("  • python run.py - 대화형 메뉴로 실행")
    else:
        print("\n⚠️ 일부 모델에 문제가 있습니다. 위 로그를 확인해주세요.")
    
    print("="*60)

if __name__ == "__main__":
    main()

