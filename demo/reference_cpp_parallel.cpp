// demo2_cpu_parallel.cpp
//
// NVIDIA 블로그의 CUDA 멀티스레드 / 멀티블록 커널 아이디어를
// "CPU 멀티스레드"로 흉내 낸 데모입니다.
// 사용자의 환경(Intel MacBook Pro 2017, NVIDIA GPU 없음)에서도
// 병렬화의 효과를 실제로 측정할 수 있도록 std::thread 만으로 구현했습니다.
//
// CUDA와의 대응 관계
//  - CUDA 의 threadIdx.x  ↔  CPU 스레드의 thread_id
//  - CUDA 의 blockDim.x   ↔  전체 스레드 개수 (stride)
//  - "grid-stride loop"   ↔  for (int i = tid; i < n; i += stride)
//
// 즉, GPU에서 수천 개의 스레드가 같은 커널을 동시에 실행하던 것을
// CPU 코어 수(예: 4개)만큼의 std::thread 로 축소해 흉내 낸 것입니다.
// 동일한 데이터 분할 전략이라는 점이 핵심입니다.
//
// 빌드:
//   clang++ -O2 -std=c++17 -pthread demo2_cpu_parallel.cpp -o demo2_cpu_parallel
// 실행:
//   ./demo2_cpu_parallel
//   ./demo2_cpu_parallel 8     # 스레드 개수 지정 (선택)

#include <iostream>
#include <math.h>
#include <chrono>
#include <thread>
#include <vector>

// CUDA 커널과 같은 모양의 "grid-stride loop"를 CPU 스레드 함수로 옮긴 것.
// tid    : 이 스레드의 ID (0..stride-1) — CUDA의 threadIdx.x 에 대응
// stride : 전체 스레드 수            — CUDA의 blockDim.x * gridDim.x 에 대응
void add_worker(int tid, int stride, int n, float *x, float *y) {
    for (int i = tid; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}

void add_parallel(int n, float *x, float *y, int num_threads) {
    std::vector<std::thread> pool;
    pool.reserve(num_threads);
    for (int t = 0; t < num_threads; t++) {
        pool.emplace_back(add_worker, t, num_threads, n, x, y);
    }
    for (auto &th : pool) {
        th.join();
    }
}

int main(int argc, char **argv) {
    const int N = 1 << 20;

    int num_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (num_threads <= 0) num_threads = 4;
    if (argc >= 2) num_threads = std::max(1, std::atoi(argv[1]));

    float *x = new float[N];
    float *y = new float[N];
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    add_parallel(N, x, y, num_threads);
    auto t1 = std::chrono::high_resolution_clock::now();

    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[demo2: parallel CPU (std::thread)]" << std::endl;
    std::cout << "N           = " << N << std::endl;
    std::cout << "num_threads = " << num_threads << std::endl;
    std::cout << "Max error   = " << maxError << std::endl;
    std::cout << "Elapsed     = " << ms << " ms" << std::endl;
    double gb = 3.0 * N * sizeof(float) / 1e9;
    std::cout << "Bandwidth   = " << (gb / (ms / 1000.0)) << " GB/s" << std::endl;

    delete[] x;
    delete[] y;
    return 0;
}
