# CUDA 입문 한국어 해설 — “An Even Easier Introduction to CUDA”

NVIDIA 공식 블로그 글
[**An Even Easier Introduction to CUDA**](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
(저자: Mark Harris, 최신 갱신 2025-05-02) 의 내용을
**한국어로 아주 자세히 풀어 쓰고**, 그 위에 GPU 가 없는 환경에서도
직접 돌려 볼 수 있는 두 개의 데모 프로그램을 얹어 비교한 학습용 저장소입니다.

> 이 저장소의 목표는 “CUDA 를 처음 보는 사람이 *왜* GPU 가 빠른지를
> 코드 한 줄, 측정값 한 줄로 납득하도록 만드는 것” 입니다.

---

## 0. 한눈에 보기

| 항목 | 내용 |
|---|---|
| 원문 | https://developer.nvidia.com/blog/even-easier-introduction-cuda/ |
| 핵심 예제 | 두 배열 더하기 `y[i] = x[i] + y[i]`, N = 2²⁰ ≈ 1,048,576 |
| 진행 단계 | CPU 단일 → CUDA 1스레드 → 1블록·256스레드 → 다중 블록 + 메모리 프리페치 |
| 본 저장소의 데모 | (1) 순수 Python 순차 (2) NumPy 벡터화 + multiprocessing 병렬 |
| 실측 환경 | MacBook Pro 2017, Intel Core i5/i7 (4 코어), macOS Ventura, Python 3 + NumPy 2.4 |

---

## 1. 디렉터리 구조

```
cuda_intro/
├── README.md                    ← (이 문서) 한국어 개요 + 원문 해설
├── comparison.md                ← 데모 1 vs 데모 2 vs CUDA 자세한 비교 페이지
└── demo/
    ├── demo1_sequential.py      ← 데모 1: 순수 Python 순차 더하기 (Mac 즉시 실행)
    ├── demo2_parallel.py        ← 데모 2: NumPy 벡터화 + multiprocessing (Mac 즉시 실행)
    ├── reference_cpp_sequential.cpp   ← 참고: 블로그 원문 C++ 순차 버전
    ├── reference_cpp_parallel.cpp     ← 참고: std::thread 로 옮긴 병렬 버전
    ├── reference_cuda.cu              ← 참고: 블로그 원문 CUDA 최종 버전
    └── Makefile                       ← 참고용 C++ 빌드(Xcode CLT 필요)
```

> **참고:** 사용자의 환경에는 NVIDIA GPU 가 없으므로 `reference_cuda.cu` 는
> 동작하지 않으며, `reference_cpp_*.cpp` 도 Xcode Command Line Tools 가 깨져 있는
> 경우 `xcode-select --install` 후에 빌드 가능합니다. 이 글에서는 **즉시 동작하는
> Python 데모** 가 메인입니다.

---

## 2. 원문 해설 — 단계별로 풀어 쓰기

### 2.1 출발점: 그냥 C++ 코드

블로그는 GPU 이야기를 시작하기 전에, 먼저 **순수 CPU C++ 코드 한 개** 를 보여 줍니다.

```cpp
void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++) y[i] = x[i] + y[i];
}

int main() {
    int N = 1<<20;
    float *x = new float[N], *y = new float[N];
    for (int i = 0; i < N; i++) { x[i] = 1.0f; y[i] = 2.0f; }
    add(N, x, y);
    /* ... 검증 ... */
}
```

핵심은 단 두 줄입니다.

* `add()` 는 **단일 스레드** 가 N 번 반복하며 한 원소씩 더합니다.
* 100만 원소면 현대 CPU 한 코어에서 수 ms ~ 수십 ms 면 끝납니다.

이 코드가 본 저장소의 [`demo/demo1_sequential.py`](demo/demo1_sequential.py) 의
원형이며, 본 데모에서는 측정값을 직접 보여 줍니다.

### 2.2 첫 CUDA 코드: `__global__` 키워드와 `<<<...>>>` 실행 구성

블로그의 첫 GPU 버전은 이 두 가지만 추가합니다.

1. 함수 앞에 `__global__` 을 붙여서 **GPU 위에서 돌아가는 커널** 임을 표시.
2. 메모리는 `cudaMallocManaged()` 로 잡아서 CPU/GPU 가 같은 포인터로 접근 (**Unified Memory**).
3. 호출은 평범한 함수 호출이 아니라 `add<<<1, 1>>>(N, x, y);` 로 **실행 구성** 을 명시.

```cpp
__global__
void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++) y[i] = x[i] + y[i];
}
...
cudaMallocManaged(&x, N*sizeof(float));
cudaMallocManaged(&y, N*sizeof(float));
...
add<<<1, 1>>>(N, x, y);     // 블록 1개, 그 블록 안 스레드 1개
cudaDeviceSynchronize();    // GPU 가 끝날 때까지 대기
```

여기서 가장 중요한 점은 — **GPU 에서 실행시켰지만 스레드는 1개뿐** 이라
사실상 “느린 GPU 코어 1개로 N번 도는 for 루프” 가 됐다는 사실입니다.
NSight Systems 프로파일러로 측정해 보면 T4 GPU 에서 약 **75 ms** 가 나옵니다.
GPU 한 코어는 CPU 한 코어보다 훨씬 느리기 때문에 베이스라인은 오히려 늦습니다.

### 2.3 한 블록 안에 256개의 스레드: SIMT 의 등장

같은 블록 안의 스레드들은 **같은 커널 코드를 같은 시점에** 실행합니다 (SIMT).
실행 구성을 `<<<1, 256>>>` 으로 바꾸고 커널은 자기 인덱스로 자기 몫만 계산합니다.

```cpp
__global__
void add(int n, float *x, float *y) {
    int index  = threadIdx.x;   // 이 스레드가 블록 안에서 몇 번째인지 (0..255)
    int stride = blockDim.x;    // 블록 안 스레드 총 개수 (256)
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}
...
add<<<1, 256>>>(N, x, y);
```

여기 들어 있는 `for (int i = index; i < n; i += stride)` 패턴이 그 유명한
**grid-stride loop** 의 단순 버전 입니다. N 이 스레드 수보다 많아도 안전하게
모든 원소를 덮어 줍니다. 측정값은 **약 4 ms (≈19× 빠름)**.

### 2.4 여러 블록을 띄워 SM 을 모두 채우기

블로그가 강조하는 핵심 한 줄은 다음과 같습니다.

> *Each SM can run multiple concurrent thread blocks, but each thread block runs on a single SM.*

따라서 “블록 1개” 만 띄우면 GPU 의 다른 SM 들은 노는 셈입니다.
블록 수를 N 에 맞춰 늘려서 모든 SM 을 채워야 합니다.

```cpp
int blockSize = 256;
int numBlocks = (N + blockSize - 1) / blockSize;   // 1<<20 / 256 = 4096
add<<<numBlocks, blockSize>>>(N, x, y);
```

커널은 **전역 인덱스** 와 **전역 stride** 를 쓰도록 갱신됩니다.

```cpp
int index  = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
```

### 2.5 마지막 한 끗: Unified Memory 프리페치

이 시점에서 측정값은 약 4.5 ms 로 별로 줄지 않습니다. 이유는
페이지가 처음 GPU 가 만질 때마다 **페이지 폴트가 일어나며 H2D/D2H 가 따로 발생**
하기 때문입니다. 해결은 한 줄 — 데이터를 미리 GPU 로 옮겨 둡니다.

```cpp
cudaMemPrefetchAsync(x, N*sizeof(float), 0, 0);
cudaMemPrefetchAsync(y, N*sizeof(float), 0, 0);
```

그러면 커널 자체는 **약 47 µs (≈1932× 빠름, 265 GB/s)** 로 떨어지며
이는 T4 의 피크 대역폭 320 GB/s 의 80% 가 넘는 수준입니다.

### 2.6 블로그가 가르치는 한 줄 요약

* GPU 가속의 절반은 *코드를 GPU 로 옮기는 것* 이지만 (`__global__`, `<<<...>>>`),
* 진짜 절반은 *충분한 블록 수* 와 *데이터 위치* 를 정성껏 챙기는 것 입니다.

---

## 3. 본 저장소의 두 데모

블로그는 NVIDIA GPU 를 전제로 하므로, GPU 가 없는 환경에서도 같은 *교훈* 을
실측으로 체험하도록 두 데모를 준비했습니다.

### 3.1 데모 1 — `demo/demo1_sequential.py`

* 블로그의 첫 단계인 **순수 CPU 단일 스레드** 와 정확히 1:1 대응.
* 표준 라이브러리만 사용 (NumPy 도 안 씀) → 가장 느린 베이스라인.
* 본 사용자 Mac 에서 **N=1<<20 기준 ≈ 86 ms** (실측).

### 3.2 데모 2 — `demo/demo2_parallel.py`

두 가지 병렬 전략을 한 파일에 묶어 자동으로 비교 측정합니다.

| 모드 | 비유 | 한 줄 설명 |
|---|---|---|
| (a) NumPy 벡터화 | CUDA `<<<1, 256>>>` (한 블록 다중 스레드, SIMT) | C 레벨 SIMD/캐시 친화 루프로 한 코어 안에서 N 원소를 한 번에 처리 |
| (b) multiprocessing + grid-stride | CUDA `<<<numBlocks, 256>>>` (다중 블록, 다중 SM) | 4 개 프로세스가 각자 `i = tid; i += stride` 인덱스 패턴으로 자기 몫 처리 |

(b) 의 `tid::stride` NumPy 슬라이싱이 정확히 GPU 의 grid-stride loop 과
동일한 “자기 인덱스만 stride 간격으로 처리” 패턴입니다.

```python
def _worker(tid, stride, n, shm_x_name, shm_y_name):
    ...
    y[tid::stride] = x[tid::stride] + y[tid::stride]   # ← grid-stride 그 자체
```

---

## 4. 실측 결과 (사용자 Mac 에서 직접 측정)

테스트 환경: MacBook Pro 13" 2017, Intel x86_64, 4 코어, macOS Ventura, Python 3 + NumPy 2.4.4.
각 값은 여러 회 실행 중 가장 빠른 시간을 채택했습니다.

| 워크로드 | 데모1 (순수 Python) | 데모2 (a) NumPy 벡터화 | 데모2 (b) multi-process |
|---|---:|---:|---:|
| N = 1<<20 (1M)  | 85.9 ms (0.15 GB/s) |  **0.78 ms (16.1 GB/s)** | 374 ms (0.03 GB/s) |
| N = 1<<24 (16M) |        —            | **27.4 ms (7.4 GB/s)**  | 564 ms (0.36 GB/s) |
| N = 1<<26 (64M) |        —            | **216 ms (3.7 GB/s)**   | 1166 ms (0.69 GB/s) |

자세한 해설과 CUDA 측정값과의 비교는 [`comparison.md`](comparison.md) 에 정리했습니다.

### 한눈에 알 수 있는 점
1. **데모1 → 데모2(a) 만 가도 ≈100×** — 인터프리트 루프 vs C 레벨 SIMD 의 차이.
2. **데모2(b) multiprocessing 은 N 이 작을 땐 오히려 손해** — 프로세스 spawn /
   shared memory 셋업 비용이 덧셈 자체보다 훨씬 큽니다. 이것이 블로그가 강조한
   *“커널이 짧으면 launch overhead 가 지배적”* 교훈의 CPU 버전입니다.
3. CUDA 가 (b) 에서 얻고자 하는 것은 “싸고 가벼운 스레드” 입니다.
   GPU 의 스레드 launch 비용은 µs 미만이라 (a) 보다 (b) 가 훨씬 유리한 영역으로
   넘어갈 수 있습니다. 본 데모에서는 그 차이를 *눈으로* 볼 수 있습니다.

---

## 5. 직접 실행하기

### 데모 (Python, 즉시 실행)

```bash
cd demo

# 데모 1: 순차
python3 demo1_sequential.py

# 데모 2: 두 가지 병렬 전략 자동 측정
python3 demo2_parallel.py
python3 demo2_parallel.py --n 16777216 --procs 4
python3 demo2_parallel.py --n 67108864 --procs 4 --repeat 3
```

### 참고 C++ 코드 빌드 (선택, Xcode CLT 필요)

```bash
xcode-select --install         # 필요시 한 번
cd demo
make all                        # build/demo1_cpu_sequential, build/demo2_cpu_parallel
make run                        # 둘 다 실행
```

### 참고 CUDA 코드 빌드 (NVIDIA GPU + CUDA Toolkit 환경에서만)

```bash
nvcc -O2 reference_cuda.cu -o reference_cuda
./reference_cuda
nsys profile --stats=true ./reference_cuda
```

---

## 6. 더 읽을거리

원문 글이 추천한 학습 자료 그대로 옮깁니다.

* NVIDIA DLI: *Getting Started with Accelerated Computing in Modern CUDA C++* (8 시간+)
* NVIDIA DLI: *Fundamentals of Accelerated Computing with CUDA Python*
* CUDA 공식 *Programming Guide* / *Best Practices Guide*
* 블로그의 후속 시리즈 — shared memory, 행렬 연산, 레이트레이싱 등

---

**라이선스 / 출처**
* 본 저장소의 코드와 한국어 해설은 학습 자료로 자유롭게 사용 가능합니다.
* 원 글, 표, 측정값(NVIDIA T4 측) 인용 출처: <https://developer.nvidia.com/blog/even-easier-introduction-cuda/>
* 본 저장소의 측정값은 사용자의 Intel Mac 에서 직접 측정한 값입니다.
