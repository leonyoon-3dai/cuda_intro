"""demo2_parallel.py

NVIDIA "An Even Easier Introduction to CUDA" 글의 GPU 병렬화 단계를
사용자의 GPU 없는 Intel Mac 에서 흉내 내고 측정하는 데모입니다.

블로그의 진행 단계:
    (1) add<<<1, 1>>>(...)            : GPU 1스레드 (≈ 75 ms on T4)
    (2) add<<<1, 256>>>(...)          : 한 블록 256스레드 (≈ 4 ms)
    (3) add<<<numBlocks, 256>>>(...)  : 여러 블록 (≈ 0.05 ms, 메모리 프리페치 시)

본 데모에서 매핑:
    (a) NumPy 벡터화        ↔  (2)  "한 블록, 여러 스레드"
        - C 레벨 SIMD/캐시 친화 루프로 N개 원소를 한 번에 처리
        - GPU의 SIMT(=같은 명령을 여러 데이터에) 비유로 가장 가까움
    (b) multiprocessing+grid-stride loop  ↔ (3) "여러 블록"
        - 프로세스(=SM 비유)를 만들고 각 프로세스가 grid-stride 패턴으로
          자기 인덱스를 처리. 코어 수만큼 진짜 병렬.

CUDA 와의 인덱싱 대응:
    tid    : 프로세스 ID 0..P-1               ↔  blockIdx.x * blockDim.x + threadIdx.x
    stride : 프로세스 개수 P                  ↔  blockDim.x * gridDim.x
    for i = tid; i < N; i += stride: ...      ↔  grid-stride loop

실행:
  python3 demo2_parallel.py
  python3 demo2_parallel.py --n 16777216 --procs 4
"""

from __future__ import annotations

import argparse
import math
import os
import time
from multiprocessing import Process, shared_memory

import numpy as np


def fmt_gbs(elapsed_s: float, n: int) -> str:
    """y = x + y 는 두 배열을 읽고 한 배열에 쓰므로 3*N*4 바이트 이동(어림)."""
    if elapsed_s <= 0:
        return "n/a"
    bytes_moved = 3 * n * 4
    return f"{bytes_moved / 1e9 / elapsed_s:6.3f} GB/s"


# ---------- (a) NumPy 벡터화 ----------------------------------------------------
def run_numpy_vectorized(n: int) -> float:
    """한 프로세스 안에서 NumPy 한 줄로 모든 원소 덧셈.

    이는 CUDA 의 add<<<1, 256>>> 처럼 "같은 명령 여러 데이터" 모델이며,
    내부에서 SIMD(예: AVX2)와 캐시 친화 루프를 활용합니다.
    """
    x = np.ones(n, dtype=np.float32)
    y = np.full(n, 2.0, dtype=np.float32)
    t0 = time.perf_counter()
    y[:] = x + y
    t1 = time.perf_counter()
    err = float(np.max(np.abs(y - 3.0)))
    if err != 0.0:
        raise RuntimeError(f"max error mismatch: {err}")
    return t1 - t0


# ---------- (b) multiprocessing + grid-stride loop -----------------------------
def _worker(tid: int, stride: int, n: int, shm_x_name: str, shm_y_name: str) -> None:
    shm_x = shared_memory.SharedMemory(name=shm_x_name)
    shm_y = shared_memory.SharedMemory(name=shm_y_name)
    try:
        x = np.ndarray((n,), dtype=np.float32, buffer=shm_x.buf)
        y = np.ndarray((n,), dtype=np.float32, buffer=shm_y.buf)
        # 정확히 CUDA 의 grid-stride loop 패턴.
        # 각 워커는 [tid, tid+stride, tid+2*stride, ...] 인덱스만 담당.
        y[tid::stride] = x[tid::stride] + y[tid::stride]
    finally:
        shm_x.close()
        shm_y.close()


def run_multiprocess_grid_stride(n: int, num_procs: int) -> float:
    """N 원소를 num_procs 개의 프로세스가 grid-stride 로 나눠서 처리."""
    x = np.ones(n, dtype=np.float32)
    y = np.full(n, 2.0, dtype=np.float32)

    shm_x = shared_memory.SharedMemory(create=True, size=x.nbytes)
    shm_y = shared_memory.SharedMemory(create=True, size=y.nbytes)
    try:
        sx = np.ndarray(x.shape, dtype=x.dtype, buffer=shm_x.buf)
        sy = np.ndarray(y.shape, dtype=y.dtype, buffer=shm_y.buf)
        sx[:] = x
        sy[:] = y

        # 측정 구간: 프로세스 시작 + 작업 + 합류 (CUDA 의 kernel launch + 실행 + sync 와 같은 의미)
        t0 = time.perf_counter()
        procs = [
            Process(target=_worker, args=(t, num_procs, n, shm_x.name, shm_y.name))
            for t in range(num_procs)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        t1 = time.perf_counter()

        err = float(np.max(np.abs(sy - 3.0)))
        if err != 0.0:
            raise RuntimeError(f"max error mismatch: {err}")
        return t1 - t0
    finally:
        shm_x.close()
        shm_x.unlink()
        shm_y.close()
        shm_y.unlink()


# ---------- 실행 ---------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="demo2: parallel CPU mirror of CUDA kernel")
    parser.add_argument("--n", type=int, default=1 << 20,
                        help="원소 개수 (기본 1<<20 = 1,048,576)")
    parser.add_argument("--procs", type=int, default=os.cpu_count() or 4,
                        help="multiprocessing 프로세스 개수 (기본 = CPU 코어 수)")
    parser.add_argument("--repeat", type=int, default=3,
                        help="각 모드 반복 횟수, 가장 빠른 시간을 채택")
    args = parser.parse_args()

    n = args.n
    p = max(1, args.procs)
    r = max(1, args.repeat)

    print(f"[demo2 settings] N = {n:,}    procs = {p}    repeat = {r}")
    print()

    # (a) NumPy 벡터화
    best_a = min(run_numpy_vectorized(n) for _ in range(r))
    print("(a) NumPy vectorized  (≈ CUDA <<<1, 256>>>, SIMT in one core)")
    print(f"    Elapsed = {best_a * 1000:8.3f} ms    {fmt_gbs(best_a, n)}")
    print()

    # (b) multiprocessing + grid-stride
    best_b = min(run_multiprocess_grid_stride(n, p) for _ in range(r))
    print(f"(b) multiprocessing + grid-stride loop  (≈ CUDA <<<numBlocks, 256>>>)")
    print(f"    Elapsed = {best_b * 1000:8.3f} ms    {fmt_gbs(best_b, n)}")
    print(f"    참고: 측정값에는 프로세스 생성/공유 메모리 셋업이 포함됩니다.")
    print(f"          (CUDA 의 kernel launch + sync 비용에 대응)")
    print()

    print("[해설]")
    if best_a < best_b:
        ratio = best_b / best_a
        print(f"이 워크로드(N={n:,})에서는 (a) NumPy 벡터화가 (b) 멀티프로세스 보다 약 {ratio:.1f}× 빠릅니다.")
        print("이유: 덧셈 자체가 매우 가벼워, 프로세스 생성/공유 메모리 셋업 비용이 계산보다 큼.")
        print("→ 블로그에서도 '커널이 짧으면 launch overhead 가 지배적' 이라는 동일한 교훈이 나옵니다.")
        print("→ N 을 크게 하거나 (`--n 16777216`) 코어 수를 늘리면 (b) 가 (a) 를 따라잡습니다.")
    else:
        ratio = best_a / best_b
        print(f"이 워크로드에서는 (b) 멀티프로세스가 (a) NumPy 벡터화 보다 약 {ratio:.1f}× 빠릅니다.")
        print("→ N 이 충분히 커서 코어 병렬화 이득이 오버헤드를 압도하는 영역입니다.")


if __name__ == "__main__":
    main()
