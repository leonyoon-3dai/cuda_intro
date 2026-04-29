"""demo1_sequential.py

NVIDIA "An Even Easier Introduction to CUDA" 글의 첫 번째 코드(CPU 단일 스레드 버전)를
파이썬으로 옮긴 데모입니다. 사용자의 Intel Mac에서 즉시 실행됩니다.

대응 관계
  - 블로그의 add() 함수            ↔  이 파일의 add()
  - 블로그의 g++ 단일 코어 실행    ↔  파이썬 단일 인터프리터 실행
  - CUDA 의 add<<<1, 1>>>(...)     ↔  본 데모 (한 개의 흐름이 N번 더함)

실행:
  python3 demo1_sequential.py
"""

from __future__ import annotations

import math
import time

N = 1 << 20  # 1,048,576 개 (블로그와 동일)


def add(n: int, x: list[float], y: list[float]) -> None:
    """블로그 원문의 add()와 동일. 단일 스레드에서 N번 반복."""
    for i in range(n):
        y[i] = x[i] + y[i]


def main() -> None:
    x = [1.0] * N
    y = [2.0] * N

    t0 = time.perf_counter()
    add(N, x, y)
    t1 = time.perf_counter()

    # 결과 검증: 모든 값이 3.0 이어야 한다
    max_error = 0.0
    for v in y:
        max_error = max(max_error, math.fabs(v - 3.0))

    elapsed_ms = (t1 - t0) * 1000.0
    # 의미상 두 배열을 읽고 한 배열에 쓰므로 3*N*4바이트가 이동했다고 본다 (float32 기준).
    moved_bytes = 3 * N * 4
    bandwidth_gb_s = moved_bytes / 1e9 / (t1 - t0)

    print("[demo1: pure-Python sequential add]")
    print(f"N           = {N}")
    print(f"Max error   = {max_error}")
    print(f"Elapsed     = {elapsed_ms:.3f} ms")
    print(f"Bandwidth*  = {bandwidth_gb_s:.3f} GB/s   (* 어림 추정)")


if __name__ == "__main__":
    main()
