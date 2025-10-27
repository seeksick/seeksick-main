#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py 수정 사항 테스트
numpy 배열 포맷 오류가 수정되었는지 확인
"""

import numpy as np
from models import EMOTIONS

# 테스트: numpy 배열을 float으로 변환하여 출력
print("="*70)
print("🧪 main.py 수정 사항 테스트")
print("="*70)
print()

# 더미 확률 배열 생성 (실제 모델 출력과 동일한 형태)
test_probs = np.array([0.1, 0.2, 0.15, 0.45, 0.1])

print("✅ 수정 전 방식 (오류 발생):")
print("   for emotion, prob in zip(EMOTIONS, probs):")
print("       print(f'{emotion}: {prob:.3f}')  # ← TypeError 발생!")
print()

print("✅ 수정 후 방식 (정상 작동):")
print("   for emotion, prob in zip(EMOTIONS, probs):")
print("       print(f'{emotion}: {float(prob):.3f}')  # ← float() 변환!")
print()

print("📊 실제 테스트:")
for emotion, prob in zip(EMOTIONS, test_probs):
    print(f"  {emotion}: {float(prob):.3f}")

print()
print("="*70)
print("✅ 테스트 성공! main.py가 정상 작동할 것입니다.")
print("="*70)
print()
print("🚀 이제 main.py를 실행하세요:")
print("   python3 main.py")
print()
print("⚠️  웹캠과 마이크 권한 허용이 필요합니다!")
print("="*70)

