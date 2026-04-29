# 데모 1 vs 데모 2 vs CUDA 자세한 비교

이 문서는 [`demo/demo1_sequential.py`](demo/demo1_sequential.py) 와
[`demo/demo2_parallel.py`](demo/demo2_parallel.py) 그리고
NVIDIA 블로그의 CUDA 코드를 한 줄 한 줄 짝지어 비교합니다.
[README](README.md) 의 “단계별 해설” 을 코드 레벨 / 측정값 레벨로
완전히 풀어 쓴 보충 페이지입니다.

---

## 0. 비교 표 한 장 요약

| 관점 | 데모1 (순수 Python) | 데모2 (a) NumPy 벡터화 | 데모2 (b) multiprocessing | C++ 순차 (참고) | C++ `std::thread`×4 (참고) | CUDA `<<<numBlocks,256>>>` (블로그) |
|---|---|---|---|---|---|---|
| 어디서 도는가 | 인터프리터 1개 | C 라이브러리 + SIMD 1코어 | 4개의 OS 프로세스, 각자 NumPy | clang `-O2`, 1 OS 스레드 | clang `-O2`, 4 OS 스레드 | NVIDIA GPU 의 수많은 스레드 |
| 병렬화 단위 | 없음 | SIMD 레인 (≈ 8 floats / 명령) | OS 프로세스 (=4) | SIMD 레인 (오토벡터화) | OS 스레드 ×4 + SIMD | warp(32 threads) × 수천 |
| 동기화 방법 | 없음 | 함수 반환 = 끝 | `Process.join()` | (단일 흐름) | `std::thread::join()` | `cudaDeviceSynchronize()` |
| 메모리 모델 | Python list | NumPy ndarray (연속 메모리) | `multiprocessing.shared_memory` (POSIX shm) | `new float[N]` | `new float[N]` (공유) | Unified Memory + 페이지 폴트/프리페치 |
| 스레드 생성 비용 | 0 | 0 | 매우 큼 (수십~수백 ms, macOS spawn) | 0 | 작음 (~수십 µs ×4) | 매우 작음 (µs) |
| grid-stride loop | ❌ | (불필요, 한 번에 통째 처리) | ✅ `y[tid::stride] = ...` | ❌ (블로그 1단계) | ✅ `for i=tid; i<n; i+=stride` (블로그 2~3단계) | ✅ `for i = idx; i < N; i += stride` |
| Mac 에서 직접 실행 | ✅ | ✅ | ✅ | ✅ (CLT 필요) | ✅ (CLT 필요) | ❌ (참고용 코드만) |
| **N=1M 실측 (이 Mac)** | 85.9 ms | 0.78 ms | 374 ms | **0.35 ms** | 0.71 ms | (T4 기준) ≈ 47 µs |
| **N=64M 실측 (이 Mac)** | — | 216 ms | 1166 ms | **57 ms** | 85 ms | — |

---

## 1. add() 한 줄 — 코드 레벨 매핑

블로그의 모든 단계는 결국 이 한 줄을 어떻게 “여러 코어로 나누느냐” 의 이야기입니다.

### 1.1 블로그 원문 CPU 버전

```cpp
void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}
```

### 1.2 데모 1 — Python 직역

[`demo/demo1_sequential.py`](demo/demo1_sequential.py)

```python
def add(n, x, y):
    for i in range(n):
        y[i] = x[i] + y[i]
```

* 의미는 100% 동일하지만 **Python 인터프리터** 가 매 반복마다
  바이트코드 디스패치, 박싱된 float 객체 생성, 리스트 인덱싱 등을 합니다.
  → C++ 대비 100~1000× 느려질 수 있고, 그래서 **베이스라인이 크게 부풀려집니다**.
* 이는 일부러 노린 효과입니다. “무지성으로 짜면 얼마나 느린가” 를 먼저 보여 줘야
  병렬화의 가치가 와 닿습니다.

### 1.3 블로그 CUDA 단일 스레드 버전

```cpp
__global__
void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}
add<<<1, 1>>>(N, x, y);
```

* 형태는 1.1 과 같지만 함수 앞 `__global__` 이 “이 함수는 GPU 코어 위에서 호출된다”
  는 의미로 바뀌었습니다.
* `<<<1, 1>>>` 는 “블록 1개, 그 블록 안 스레드 1개” — 즉 GPU 한 코어 한 가닥에서만 실행.
* GPU 한 코어는 CPU 한 코어보다 느리므로 **약 75 ms (T4)** — 이 예제에서 가장 느린 GPU 형태.

### 1.4 블로그 CUDA `<<<1, 256>>>` 버전 (한 블록, 다중 스레드)

```cpp
__global__ void add(int n, float *x, float *y) {
    int index  = threadIdx.x;
    int stride = blockDim.x;             // = 256
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}
add<<<1, 256>>>(N, x, y);
```

* 블록 안 256 개의 스레드가 같은 커널 코드를 동시에 실행하면서
  각자 `threadIdx.x` 만큼 어긋난 인덱스부터 시작 → 256 칸씩 점프.
* 이 “**stride = blockDim.x** ; for(i=index; i<n; i+=stride)” 패턴이
  본 저장소 데모2 (b) 의 NumPy 슬라이싱과 1:1 대응.

### 1.5 데모 2 (a) — NumPy 벡터화

[`demo/demo2_parallel.py`](demo/demo2_parallel.py) 의 `run_numpy_vectorized()`

```python
y[:] = x + y
```

* 한 줄 안에서 NumPy 가 내부적으로 `for (i=0;i<N;i++)` 의 C 루프를 돌립니다.
  AVX2 같은 SIMD 까지 동원하므로 **한 코어 안에서도 8 floats / 명령** 같은 식으로
  실질적인 병렬이 일어납니다.
* CUDA 의 SIMT 와의 비유 — “같은 명령을 여러 데이터에 동시에” 라는 점은 같습니다.
  단지 차원이 8 vs. 수만 일 뿐.

### 1.6 데모 2 (b) — multiprocessing + grid-stride

```python
def _worker(tid, stride, n, shm_x_name, shm_y_name):
    ...
    y[tid::stride] = x[tid::stride] + y[tid::stride]
```

| CUDA | 데모2 (b) |
|---|---|
| `int idx    = blockIdx.x * blockDim.x + threadIdx.x;` | `tid` (프로세스 번호) |
| `int stride = blockDim.x * gridDim.x;` | `stride` (= num_procs) |
| `for (i = idx; i < N; i += stride) ...` | `y[tid::stride] = x[tid::stride] + y[tid::stride]` |

NumPy 의 슬라이싱 `tid::stride` 는 *“tid 번째부터 시작해 stride 간격으로 모든 인덱스”*
를 만들어 주므로, GPU 의 grid-stride loop 과 정확히 같은 인덱스 집합 을 처리합니다.

### 1.7 블로그 CUDA `<<<numBlocks, 256>>>` 최종 버전

```cpp
__global__ void add(int n, float *x, float *y) {
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}
int blockSize = 256;
int numBlocks = (N + blockSize - 1) / blockSize;     // 4096 for N=1M
add<<<numBlocks, blockSize>>>(N, x, y);
cudaMemPrefetchAsync(x, N*sizeof(float), 0, 0);      // ★ 메모리 위치 정리
cudaMemPrefetchAsync(y, N*sizeof(float), 0, 0);
```

* 위와 인덱싱 패턴은 동일. 차이는 **블록 수를 N 에 맞게 키워 SM 들을 모두 활용** 한다는 점.
* `cudaMemPrefetchAsync` 가 빠지면 기록상 4.5 ms / 80 GB/s 정도였는데, 넣으면 47 µs / 265 GB/s.

---

## 2. 메모리 모델 — “포인터 한 개” 의 의미가 다 다르다

### 2.1 데모 1 — Python list

```python
x = [1.0] * N
```

* `x` 는 `PyObject*` 의 배열. 각 원소는 박싱된 float 객체.
* CPU 캐시 친화도가 매우 낮고, 인덱싱마다 객체 dereference 가 발생.

### 2.2 데모 2 — NumPy ndarray + POSIX shared memory

```python
shm_x = shared_memory.SharedMemory(create=True, size=x.nbytes)
sx = np.ndarray(x.shape, dtype=x.dtype, buffer=shm_x.buf)
sx[:] = x
```

* `sx` 는 **연속된 float32 버퍼**. AVX2 SIMD 명령이 직접 들이 칠 수 있습니다.
* `shared_memory` 는 macOS/Linux 의 POSIX shm 위에 만들어지므로
  자식 프로세스들이 **데이터 복사 없이** 같은 포인터로 접근 — CUDA Unified Memory 와 비슷한 정신.

### 2.3 CUDA — Unified Memory + 페이지 폴트 + 프리페치

```cpp
cudaMallocManaged(&x, N*sizeof(float));
cudaMemPrefetchAsync(x, N*sizeof(float), 0, 0);   // 디바이스 0 으로
```

* 같은 포인터를 CPU/GPU 양쪽이 사용 가능. 단, 처음 만질 때 **페이지 폴트** 가 발생해
  H2D / D2H 가 자동 트리거됨.
* `cudaMemPrefetchAsync` 로 폴트를 미리 “예방 접종” 하면 커널 시간만 측정에 남습니다.

| 측면 | 데모1 list | 데모2 ndarray (한 프로세스) | 데모2 shm (멀티프로세스) | CUDA Unified Memory |
|---|---|---|---|---|
| 데이터 레이아웃 | 비연속 (객체 포인터) | 연속 float32 | 연속 float32 (POSIX shm) | 연속 float32 (페이지 단위 마이그레이션) |
| 캐시 친화 | 매우 나쁨 | 매우 좋음 | 매우 좋음 | 페이지 단위로 좋음 |
| 복사 비용 | (없음, 같은 프로세스) | (없음) | 0-copy 공유 | 페이지 폴트 / 프리페치 비용 |

---

## 3. 동기화

| 모델 | 데모1 | 데모2 (a) | 데모2 (b) | CUDA |
|---|---|---|---|---|
| 시작 | 함수 호출 | 함수 호출 | `Process.start()` | `add<<<...>>>(...)` (비동기) |
| 끝 대기 | (필요 없음) | (필요 없음, 동기) | `for p in procs: p.join()` | `cudaDeviceSynchronize()` |
| 실패시 | Python 예외 | Python 예외 | 자식의 예외는 메인까지 자동 전파 안 됨 (주의) | `cudaGetLastError()` |

CUDA 커널 호출은 “돌아왔다 = 시작했다” 일 뿐, **끝났다는 보장이 없다** 는 점이
초보가 가장 자주 걸리는 함정입니다. 데모2 (b) 의 `Process.join()` 이 정확히 이 역할 — 그래서
이 비유는 학습용으로 잘 들어맞습니다.

---

## 4. 측정값 — 사용자 Mac 실측 + 블로그 T4 인용

### 4.1 본 사용자 Mac (Intel x86_64, 4 cores, Apple clang 14.0.3 / Python 3 + NumPy 2.4)

| 모드 | N | Elapsed | 추정 대역폭 | vs. 데모1(N=1M) |
|---|---:|---:|---:|---:|
| 데모1 (순수 Python)              | 1,048,576    |  85.883 ms |   0.147 GB/s |  1×       |
| 데모2 (a) NumPy 벡터화           | 1,048,576    |   0.781 ms |  16.117 GB/s |  ≈ 110× 빠름 |
| 데모2 (b) MP + grid-stride (4p)  | 1,048,576    | 374.084 ms |   0.034 GB/s |  ≈ 0.23× (느림) |
| **C++ 순차 (clang -O2, 1 코어)** | 1,048,576    |   **0.348 ms** | **35.86 GB/s** | **≈ 247× 빠름** |
| C++ 병렬 (`std::thread` ×4)      | 1,048,576    |   0.715 ms |  17.61 GB/s  |  ≈ 120× 빠름 (병렬보다 순차가 빠름!) |
| 데모2 (a)                        | 16,777,216   |  27.357 ms |   7.359 GB/s |  —          |
| 데모2 (b) (4p)                   | 16,777,216   | 564.406 ms |   0.357 GB/s |  —          |
| C++ 순차                          | 16,777,216   |  14.524 ms |  13.86 GB/s  |  —          |
| C++ 병렬 (4t)                    | 16,777,216   |  15.696 ms |  12.83 GB/s  |  —          |
| 데모2 (a)                        | 67,108,864   | 216.199 ms |   3.725 GB/s |  —          |
| 데모2 (b) (4p)                   | 67,108,864   | 1165.878 ms|   0.691 GB/s |  —          |
| C++ 순차                          | 67,108,864   |  57.012 ms |  14.13 GB/s  |  —          |
| C++ 병렬 (4t)                    | 67,108,864   |  85.378 ms |   9.43 GB/s  |  —          |

> 모든 값은 같은 측정 스크립트/바이너리로 얻은 값이며, 각 모드의 *가장 빠른* 시간을 채택했습니다.
> 측정 변동(±)이 있을 수 있으니 절대값보다는 **자릿수 차이** 에 주목하세요.

#### 4.1.1 N=1M 의 흥미로운 점 — “캐시에 들어갈 때만 36 GB/s가 보인다”

C++ 순차가 N=1M 에서 35.9 GB/s 인데 N=16M 에선 13.9 GB/s 로 떨어집니다.
1M × 4바이트 × 2배열 = **8 MB** 라 본 i7 의 **L3 (8 MB 내외)** 안에 들어가지만,
16M 부터는 캐시를 벗어나 DDR4 본체 대역폭에 의존하게 됩니다. 즉,
*“진짜 실용적 워크로드의 베이스라인은 13–14 GB/s”* 입니다 — 이것이 이 Mac 에서
단순 add 가 닿을 수 있는 *상한선* 입니다.

#### 4.1.2 “4 스레드가 더 느리다” 가 보내는 메시지

`std::thread` ×4 가 모든 N 에서 1 스레드보다 느린 이유:

- **N=1M**: 캐시 안이라 메모리 대역폭에 여유가 있지만, 0.35 ms 짜리 작업에 4번의
  `pthread_create + join` 이 이미 더 큽니다 → launch overhead.
- **N≥16M**: 1 스레드가 이미 DDR4 한계에 닿아 있으므로 4 스레드는 **같은 버스를 4 명이 다투기만**
  할 뿐 더 빨라질 메모리 대역이 없습니다 → 오히려 false sharing 가능성과 컨텍스트 스위치로 손해.

블로그가 GPU 에서 보이는 “블록 수 늘리니 1932× 빨라짐” 은 GPU 가
**메모리 대역폭 자체가 한 자리 더 큰 디바이스** 이기 때문에 가능합니다. 같은 코드를
CPU 에 4 코어로 옮긴다고 그 마법이 따라오진 않습니다.

### 4.2 블로그 인용 — NVIDIA T4 GPU

| 단계 | 코드 | Elapsed | 대역폭 | vs. `<<<1,1>>>` |
|---|---|---:|---:|---:|
| ① 단일 GPU 스레드          | `add<<<1, 1>>>` | 91.8 ms | 0.137 GB/s | 1×       |
| ② 한 블록 256 스레드       | `add<<<1, 256>>>` |  2.05 ms | 6.0 GB/s   | 45×      |
| ③ 다중 블록 (no prefetch) | `add<<<numBlocks, 256>>>` |  4.5 ms | (메모리 폴트로 손실)  | —        |
| ④ ③ + 프리페치             | `cudaMemPrefetchAsync` 추가 | **47.5 µs** | **265 GB/s** | **≈ 1932×** |

*(원문 표를 인용. 본 저장소에서는 동일 GPU 가 없어 직접 재측정하지 않음.)*

### 4.3 두 표를 같이 두면 보이는 것

* **데모1 → 데모2(a)** 의 110× 점프는 블로그의 **① → ②** 의 45× 점프와 같은 종류의
  현상 — “SIMT/SIMD 가 한 *코어* 안에서 얻어 주는 이득”.
* **데모2(b)** 와 **C++ 4 스레드** 가 우리 환경에서 거의 항상 “단일 코어 + 벡터화” 보다
  느린 이유는 명확합니다 — 프로세스/스레드 셋업 비용 + 메모리 대역폭 한계.
  macOS Python 의 spawn 모드는 특히 느립니다. 이 비용은 GPU 의 kernel launch 비용 (수 µs) 과 비교되지 않을 정도로 큽니다.
* 즉, 본 데모는 “*CPU 에선 코어 수가 적어 병렬 비용이 보상받기 힘들다 → 그래서 GPU 가 필요하다*”
  는 블로그 메시지를 **부정적인 측정값** 으로 직접 보여 줍니다.
* **GPU 가 정말 빠른 핵심 이유:** 메모리 대역폭. T4 의 320 GB/s 는 본 Mac 의 DDR4
  (~25–30 GB/s) 보다 *한 자리 큰* 차원입니다. 블로그의 47 µs / 265 GB/s 와 본 Mac 의
  베스트 0.35 ms / 36 GB/s 의 차이 (약 7×) 는 거의 *대역폭 차이의 결과* 라고 봐도 됩니다.

---

## 5. 어디까지 비유가 맞고, 어디서부터 비유가 깨지나

| 블로그가 강조하는 점 | CUDA 의 실제 메커니즘 | 데모 2 (b) 의 비유 | 비유의 한계 |
|---|---|---|---|
| 같은 코드, 다른 인덱스 | 같은 SASS, threadIdx 다름 | 같은 함수, tid 다름 | OK |
| 수천 개의 스레드 | 수만~수십만 동시 실행 | 4 프로세스 | **수천 vs. 4** — 규모 차이가 큼 |
| 스레드 생성 비용이 거의 0 | warp scheduler 가 즉시 dispatch | OS 프로세스 spawn | **비용이 정반대** — CPU에선 큼 |
| Unified Memory + 프리페치 | 페이지 단위 마이그레이션 | POSIX shm | shm 은 마이그레이션 없음 |
| SM 별 L1/shared memory | 명시적 `__shared__` | (해당 없음) | 비유로는 표현 못 함 |
| 메모리 대역폭이 핵심 KPI | 320 GB/s 까지 활용 | DDR4 ~25 GB/s 한계 | 절대값 자체가 다른 차원 |

비유가 가장 **잘 맞는** 부분: 인덱싱 패턴(grid-stride), 동기화의 의미(`join`/`Synchronize`),
짧은 커널일수록 launch overhead 의 비율이 커진다는 *법칙* 자체.

비유가 **깨지는** 부분: 스레드 생성 비용, 메모리 대역폭의 절대치, 워프 단위 SIMT 의 폭,
SM 별 shared memory.

---

## 6. 직접 해 볼 수 있는 작은 실험들

블로그의 “Exercises” 를 본 데모에 맞춰 변형:

1. `python3 demo2_parallel.py --procs 1` — 프로세스 1개로 두면 (b) 가 (a) 와 비슷해야 하는지?
   결과는 **여전히 (b) 가 훨씬 느림** 입니다. 왜? → spawn + shm 셋업 자체가 고정비.
2. `--procs 8` 처럼 코어 수보다 많은 값을 줘 보세요. 시간이 어떻게 변하는지
   (블로그가 말하는 “블록 수 vs. SM 수” 관계의 CPU 버전).
3. demo1 의 `for i in range(N): ...` 안에 `print(i, threadIdx?)` 를 흉내 내고 싶으면,
   데모2 (b) 의 `_worker` 안에 `print(tid, stride)` 를 한 줄 넣어 “같은 코드, 다른 tid”
   를 눈으로 확인할 수 있습니다 (블로그 Exercise 2 의 CPU 버전).
4. demo2 의 `y[tid::stride] = x[tid::stride] + y[tid::stride]` 를 일부러
   `y[tid*chunk:(tid+1)*chunk]` 같이 “블록 분할” 로 바꿔 비교해 보세요. 캐시 친화도와
   부하 균형의 trade-off 를 체감할 수 있습니다 (CUDA 에서도 같은 trade-off 가 있음).

---

## 7. 결론 한 줄

> **“느린 base 를 깔고, 같은 코드를 인덱스만 다르게 여러 코어가 돌리는 것 — 이것이 CUDA 의 핵심.”**
>
> 본 저장소의 데모1·데모2 와 블로그의 단계는 정확히 같은 *발상* 을, 서로 다른 하드웨어 위에서
> 보여 줍니다. CPU 에서는 인덱싱 패턴은 흉내 낼 수 있어도 하드웨어가 가진 *수* 와 *대역폭* 까지
> 흉내 내지는 못합니다. 그래서 우리는 GPU 가 필요합니다.
