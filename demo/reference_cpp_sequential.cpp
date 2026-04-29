// demo1_cpu_sequential.cpp
//
// NVIDIA "An Even Easier Introduction to CUDA" 글의 첫 번째 코드(순수 CPU 버전)를
// 그대로 재현한 데모입니다. 1M(1<<20) 개의 float 원소를 가진 두 배열을 더합니다.
//
// 핵심 포인트
//  - add() 함수는 단일 스레드에서 for 루프로 N번 실행됩니다.
//  - GPU나 CUDA가 없는 Intel Mac에서도 그대로 컴파일·실행됩니다.
//  - CUDA 버전과 비교할 "기준선(baseline)" 역할을 합니다.
//
// 빌드:
//   clang++ -O2 -std=c++17 reference_cpp_sequential.cpp -o reference_cpp_sequential
// 실행:
//   ./reference_cpp_sequential                # N = 1<<20
//   ./reference_cpp_sequential 24             # N = 1<<24

#include <iostream>
#include <math.h>
#include <chrono>
#include <cstdlib>

// 두 배열의 원소를 모두 더해 y에 저장한다 (블로그 원문과 동일).
void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}

int main(int argc, char **argv) {
    int log2N = 20;
    if (argc >= 2) log2N = std::atoi(argv[1]);
    const int N = 1 << log2N;

    float *x = new float[N];
    float *y = new float[N];

    // 호스트(CPU) 메모리에서 배열 초기화
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // 시간 측정: CPU에서 add() 1회 실행
    auto t0 = std::chrono::high_resolution_clock::now();
    add(N, x, y);
    auto t1 = std::chrono::high_resolution_clock::now();

    // 결과 검증: 모든 값이 3.0f이어야 함
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[demo1: sequential CPU]" << std::endl;
    std::cout << "N           = " << N << std::endl;
    std::cout << "Max error   = " << maxError << std::endl;
    std::cout << "Elapsed     = " << ms << " ms" << std::endl;
    // 메모리 대역폭 어림: 두 배열을 읽고 한 배열에 쓰므로 3*N*sizeof(float) 바이트가 이동.
    double gb = 3.0 * N * sizeof(float) / 1e9;
    std::cout << "Bandwidth   = " << (gb / (ms / 1000.0)) << " GB/s" << std::endl;

    delete[] x;
    delete[] y;
    return 0;
}
