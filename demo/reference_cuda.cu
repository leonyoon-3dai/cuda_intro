// reference_cuda.cu
//
// NVIDIA 블로그 "An Even Easier Introduction to CUDA"의 최종 버전(멀티블록 + 프리페치).
// 본 파일은 NVIDIA GPU + CUDA Toolkit이 있는 환경에서만 컴파일 / 실행됩니다.
// (사용자의 Intel Mac 환경에서는 실행되지 않습니다. 학습/비교용 참조 코드입니다.)
//
// 빌드 (Linux / Windows + NVIDIA GPU):
//   nvcc -O2 reference_cuda.cu -o reference_cuda
// 실행:
//   ./reference_cuda
//   nsys profile --stats=true ./reference_cuda    # 프로파일링

#include <iostream>
#include <math.h>

// CUDA 커널: GPU 위에서 실행됨을 알리는 __global__ 지정자.
// 실행 구성 <<<numBlocks, blockSize>>> 에 따라
// numBlocks * blockSize 개의 스레드가 이 함수를 동시에 실행한다.
__global__
void add(int n, float *x, float *y) {
    // 전역 스레드 인덱스
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    // 전체 스레드 수 = grid 안에 있는 모든 스레드 개수
    int stride = blockDim.x * gridDim.x;
    // grid-stride loop : N이 스레드 수보다 커도 안전하게 처리
    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}

int main() {
    int N = 1 << 20;
    float *x, *y;

    // 통합 메모리(Unified Memory) — CPU/GPU 양쪽에서 같은 포인터로 접근 가능
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // 페이지를 GPU로 미리 옮겨 H2D 페이지 폴트를 줄인다.
    int device = 0;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(x, N * sizeof(float), device, 0);
    cudaMemPrefetchAsync(y, N * sizeof(float), device, 0);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);

    // GPU가 끝날 때까지 대기 (kernel launch는 비동기)
    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);
    return 0;
}
