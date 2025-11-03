#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Late Fusion 기능 테스트
5초 간격으로 멀티모달 감정을 통합하는 시뮬레이션
"""

import numpy as np
import time
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

def test_late_fusion():
    """Late Fusion 테스트"""
    print("="*80)
    print(f"{Colors.BOLD}🧪 Late Fusion 테스트 시뮬레이션{Colors.END}")
    print("="*80)
    print()
    print("5초 동안 여러 모달리티의 결과를 수집한 후 통합합니다.")
    print()
    
    # 시뮬레이션: 5초 동안의 결과 수집
    print("📊 5초 동안 결과 수집 중...")
    print()
    
    # 시나리오 1: 행복한 감정 (모든 모달리티)
    print("시나리오 1: 행복한 감정")
    print("-" * 80)
    
    face_results = [
        np.array([0.7, 0.1, 0.1, 0.05, 0.05]),  # happy 우세
        np.array([0.8, 0.05, 0.1, 0.03, 0.02]), # happy 강함
        np.array([0.75, 0.08, 0.12, 0.03, 0.02]) # happy 우세
    ]
    
    voice_results = [
        np.array([0.6, 0.15, 0.15, 0.05, 0.05])  # happy 우세
    ]
    
    text_results = [
        np.array([0.9, 0.03, 0.04, 0.02, 0.01])  # happy 매우 강함
    ]
    
    # Late Fusion 계산
    all_results = []
    
    if face_results:
        face_avg = np.mean(face_results, axis=0)
        all_results.append(face_avg)
        print(f"👤 얼굴 평균: {face_avg}")
    
    if voice_results:
        voice_avg = np.mean(voice_results, axis=0)
        all_results.append(voice_avg)
        print(f"🎤 음성 평균: {voice_avg}")
    
    if text_results:
        text_avg = np.mean(text_results, axis=0)
        all_results.append(text_avg)
        print(f"📝 텍스트 평균: {text_avg}")
    
    # 최종 융합
    fused = np.mean(all_results, axis=0)
    max_idx = np.argmax(fused)
    final_emotion = EMOTIONS[max_idx]
    confidence = float(fused[max_idx])
    
    print()
    print(f"🔀 Late Fusion 결과:")
    for emotion, prob in zip(EMOTIONS, fused):
        bar = "█" * int(prob * 40)
        print(f"   {emotion:12s} {prob:.3f} {bar}")
    
    print()
    print(f"{Colors.BOLD}{Colors.RED}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.RED}   🏆 최종 감정: {final_emotion.upper()}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.RED}   📈 신뢰도: {confidence:.1%}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.RED}{'='*80}{Colors.END}")
    print()
    
    # 시나리오 2: 우울한 감정 (일부 모달리티만)
    print("\n" + "="*80)
    print("시나리오 2: 우울한 감정 (텍스트와 음성만)")
    print("-" * 80)
    
    voice_results2 = [
        np.array([0.1, 0.7, 0.1, 0.05, 0.05]),
        np.array([0.08, 0.75, 0.12, 0.03, 0.02])
    ]
    
    text_results2 = [
        np.array([0.05, 0.85, 0.05, 0.03, 0.02])
    ]
    
    all_results2 = []
    modalities = []
    
    if voice_results2:
        voice_avg = np.mean(voice_results2, axis=0)
        all_results2.append(voice_avg)
        modalities.append(f"음성 ({len(voice_results2)}개)")
        print(f"🎤 음성 평균: {voice_avg}")
    
    if text_results2:
        text_avg = np.mean(text_results2, axis=0)
        all_results2.append(text_avg)
        modalities.append(f"텍스트 ({len(text_results2)}개)")
        print(f"📝 텍스트 평균: {text_avg}")
    
    fused2 = np.mean(all_results2, axis=0)
    max_idx2 = np.argmax(fused2)
    final_emotion2 = EMOTIONS[max_idx2]
    confidence2 = float(fused2[max_idx2])
    
    print()
    print(f"📊 사용된 모달리티: {', '.join(modalities)}")
    print()
    print(f"🔀 Late Fusion 결과:")
    for emotion, prob in zip(EMOTIONS, fused2):
        bar = "█" * int(prob * 40)
        print(f"   {emotion:12s} {prob:.3f} {bar}")
    
    print()
    print(f"{Colors.BOLD}{Colors.RED}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.RED}   🏆 최종 감정: {final_emotion2.upper()}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.RED}   📈 신뢰도: {confidence2:.1%}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.RED}{'='*80}{Colors.END}")
    print()
    
    print("="*80)
    print("✅ Late Fusion 테스트 완료!")
    print("="*80)
    print()
    print("💡 main.py를 실행하면 실제로 5초마다 이런 방식으로 통합됩니다:")
    print("   python3 main.py")
    print()

if __name__ == "__main__":
    test_late_fusion()

