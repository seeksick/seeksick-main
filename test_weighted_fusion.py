#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
가중 평균 Late Fusion 테스트
정확도 기반 가중치 적용 및 5가지 감정 확률 모두 출력
"""

import numpy as np
from models import EMOTIONS

# ANSI 색상 코드
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def test_weighted_fusion():
    """가중 평균 Late Fusion 테스트"""
    print("="*80)
    print(f"{Colors.BOLD}🧪 가중 평균 Late Fusion 테스트{Colors.END}")
    print("="*80)
    print()
    
    # 가중치 계산
    accuracies = {'face': 74.0, 'voice': 65.0, 'text': 66.0}
    total_acc = sum(accuracies.values())
    
    # 성능 비례 가중치
    perf_weights = {k: v/total_acc for k, v in accuracies.items()}
    
    # 균등 가중치
    equal_weight = 1.0 / 3.0
    
    # 하이브리드 (α = 0.6)
    alpha = 0.6
    weights = {
        'face': alpha * perf_weights['face'] + (1-alpha) * equal_weight,
        'voice': alpha * perf_weights['voice'] + (1-alpha) * equal_weight,
        'text': alpha * perf_weights['text'] + (1-alpha) * equal_weight
    }
    
    print(f"⚖️  가중치 설정:")
    print(f"   👤 얼굴 (74% 정확도): {weights['face']:.3f}")
    print(f"   🎤 음성 (65% 정확도): {weights['voice']:.3f}")
    print(f"   📝 텍스트 (66% 정확도): {weights['text']:.3f}")
    print(f"   합계: {sum(weights.values()):.3f}")
    print()
    
    # 시나리오 1: 행복한 감정
    print("="*80)
    print("📝 시나리오 1: 행복한 감정")
    print("="*80)
    print()
    
    # 5초 동안 수집된 결과
    face_results = [
        np.array([0.7, 0.1, 0.1, 0.05, 0.05]),
        np.array([0.8, 0.05, 0.1, 0.03, 0.02]),
        np.array([0.75, 0.08, 0.12, 0.03, 0.02])
    ]
    voice_results = [np.array([0.6, 0.15, 0.15, 0.05, 0.05])]
    text_results = [np.array([0.9, 0.03, 0.04, 0.02, 0.01])]
    
    # 모달리티별 평균
    face_avg = np.mean(face_results, axis=0)
    voice_avg = np.mean(voice_results, axis=0)
    text_avg = np.mean(text_results, axis=0)
    
    print(f"👤 얼굴 평균 ({len(face_results)}개):")
    for emo, prob in zip(EMOTIONS, face_avg):
        print(f"   {emo:12s} {prob:.3f}")
    
    print(f"\n🎤 음성 평균 ({len(voice_results)}개):")
    for emo, prob in zip(EMOTIONS, voice_avg):
        print(f"   {emo:12s} {prob:.3f}")
    
    print(f"\n📝 텍스트 평균 ({len(text_results)}개):")
    for emo, prob in zip(EMOTIONS, text_avg):
        print(f"   {emo:12s} {prob:.3f}")
    
    # 가중 평균 Late Fusion
    fused = (face_avg * weights['face'] + 
             voice_avg * weights['voice'] + 
             text_avg * weights['text'])
    
    # 정규화
    total_weight = weights['face'] + weights['voice'] + weights['text']
    fused /= total_weight
    
    print(f"\n{Colors.BOLD}{Colors.RED}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.RED}🎯 가중 평균 Late Fusion 결과:{Colors.END}")
    print(f"{Colors.BOLD}{Colors.RED}{'='*80}{Colors.END}")
    
    # 이모지 매핑
    emotion_emojis = {
        'happy': '😊',
        'depressed': '😢',
        'surprised': '😮',
        'angry': '😠',
        'neutral': '😐'
    }
    
    # 확률 순서대로 정렬
    sorted_indices = np.argsort(fused)[::-1]
    
    for idx in sorted_indices:
        emo = EMOTIONS[idx]
        prob = float(fused[idx])
        emoji = emotion_emojis.get(emo, '❓')
        
        # 바 그래프 (40칸)
        bar_length = int(prob * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        
        # 최고 확률이면 빨간색
        if idx == sorted_indices[0]:
            print(f"{Colors.BOLD}{Colors.RED}   {emoji} {emo:12s} [{bar}] {prob:.1%} ⭐{Colors.END}")
        else:
            print(f"   {emoji} {emo:12s} [{bar}] {prob:.1%}")
    
    max_idx = np.argmax(fused)
    final_emotion = EMOTIONS[max_idx]
    confidence = float(fused[max_idx])
    
    print(f"\n{Colors.BOLD}{Colors.RED}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.RED}   🏆 최종 감정: {final_emotion.upper()} (신뢰도: {confidence:.1%}){Colors.END}")
    print(f"{Colors.BOLD}{Colors.RED}{'='*80}{Colors.END}")
    
    # 확률 합 검증
    print(f"\n✅ 확률 합: {np.sum(fused):.6f} (1.0이어야 함)")
    
    # 시나리오 2: 우울한 감정
    print("\n\n" + "="*80)
    print("📝 시나리오 2: 우울한 감정 (음성 + 텍스트만)")
    print("="*80)
    print()
    
    voice_results2 = [
        np.array([0.1, 0.7, 0.1, 0.05, 0.05]),
        np.array([0.08, 0.75, 0.12, 0.03, 0.02])
    ]
    text_results2 = [np.array([0.05, 0.85, 0.05, 0.03, 0.02])]
    
    voice_avg2 = np.mean(voice_results2, axis=0)
    text_avg2 = np.mean(text_results2, axis=0)
    
    print(f"🎤 음성 평균 ({len(voice_results2)}개):")
    for emo, prob in zip(EMOTIONS, voice_avg2):
        print(f"   {emo:12s} {prob:.3f}")
    
    print(f"\n📝 텍스트 평균 ({len(text_results2)}개):")
    for emo, prob in zip(EMOTIONS, text_avg2):
        print(f"   {emo:12s} {prob:.3f}")
    
    # 가중 평균 (얼굴 없음)
    fused2 = (voice_avg2 * weights['voice'] + 
              text_avg2 * weights['text'])
    total_weight2 = weights['voice'] + weights['text']
    fused2 /= total_weight2
    
    print(f"\n{Colors.BOLD}{Colors.RED}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.RED}🎯 가중 평균 Late Fusion 결과 (얼굴 없음):{Colors.END}")
    print(f"{Colors.BOLD}{Colors.RED}{'='*80}{Colors.END}")
    
    sorted_indices2 = np.argsort(fused2)[::-1]
    
    for idx in sorted_indices2:
        emo = EMOTIONS[idx]
        prob = float(fused2[idx])
        emoji = emotion_emojis.get(emo, '❓')
        
        bar_length = int(prob * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        
        if idx == sorted_indices2[0]:
            print(f"{Colors.BOLD}{Colors.RED}   {emoji} {emo:12s} [{bar}] {prob:.1%} ⭐{Colors.END}")
        else:
            print(f"   {emoji} {emo:12s} [{bar}] {prob:.1%}")
    
    max_idx2 = np.argmax(fused2)
    final_emotion2 = EMOTIONS[max_idx2]
    confidence2 = float(fused2[max_idx2])
    
    print(f"\n{Colors.BOLD}{Colors.RED}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.RED}   🏆 최종 감정: {final_emotion2.upper()} (신뢰도: {confidence2:.1%}){Colors.END}")
    print(f"{Colors.BOLD}{Colors.RED}{'='*80}{Colors.END}")
    
    print(f"\n✅ 확률 합: {np.sum(fused2):.6f}")
    
    print("\n" + "="*80)
    print("✅ 가중 평균 Late Fusion 테스트 완료!")
    print("="*80)
    print()
    print("💡 main.py를 실행하면 실제로 이 방식으로 통합됩니다:")
    print("   python3 main.py")
    print()

if __name__ == "__main__":
    test_weighted_fusion()

