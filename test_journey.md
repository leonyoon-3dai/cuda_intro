# 테스트 진행 기록 — Xcode CLT 설치부터 C++ 측정·분석까지

이 문서는 [README](README.md) 와 [comparison.md](comparison.md) 에 들어간
**C++ 참고 데모의 측정값** 을 어떻게 얻었는지를 그대로 재현 가능한 형태로 남긴
*실험 노트* 입니다. 처음 막혔던 부분, 시도해 보고 버린 우회로, 최종적으로 통한 길,
그리고 결과 해석까지를 시간 순서대로 기록합니다.

> 환경: MacBook Pro 13" 2017, Intel Core i5/i7 4 코어, macOS Ventura 13(22.6.0),
> RAM 16GB, NVIDIA GPU 없음, Python 3 (miniconda3) + NumPy 2.4.4.

---

## 0. 한 줄 결론

> Apple clang 14 로 빌드한 C++ 순차 add 가 N=1M 에서 **0.35 ms / 35.9 GB/s** 로
> 측정되었고, `std::thread` 4 개로 늘리면 **오히려 더 느렸다**.
> 즉 본 워크로드는 *memory-bandwidth-bound* 라 CPU 4코어로는 더 이상 빨라지지
> 않으며, **이것이 GPU(예: T4, 320 GB/s)가 의미 있는 진짜 이유**.

---

## 1. 시작 상황 — 어디서 막혔는가

처음에 단순히 `make all` 을 시도하자 다음 에러가 났습니다.

```
$ make all
xcrun: error: invalid active developer path (/Library/Developer/CommandLineTools),
       missing xcrun at: /Library/Developer/CommandLineTools/usr/bin/xcrun
make: *** [reference_cpp_sequential] Error 1
```

확인해 보니 `xcode-select -p` 는 `/Library/Developer/CommandLineTools` 를 가리키는데,
실제로 그 안의 `usr` 디렉터리가 비어 있었습니다.

```
$ ls /Library/Developer/CommandLineTools/usr/
share
```

— 즉, Command Line Tools 의 메타데이터만 남고 **실제 컴파일러 바이너리 자체가 없는**
상태였습니다 (이전 macOS 업그레이드 등으로 깨진 것으로 추정).

`/usr/bin/clang++`, `/usr/bin/g++` 는 모두 “바로가기” 만 있고 실제 호출은 CLT 의
`xcrun` 을 거쳐 가므로, 이 상태에서는 **사실상 컴파일러가 한 개도 없는 상태**.

---

## 2. 우회로 탐색 (그리고 포기)

본격적으로 Xcode CLT 를 다시 깔기 전에, 다음 우회로를 시도했고 모두 막혔습니다.

### 2.1 Homebrew 의 GCC/LLVM ?

```
$ brew --version
Homebrew 5.1.5
$ brew list --formula | grep -E 'gcc|llvm|clang'
(아무 것도 없음)
```

설치된 brew 컴파일러가 없었습니다. 새로 깔려면 결국 `brew install gcc` 또는
`llvm` 인데, 그것 자체가 *컴파일러* 를 요구하는 상황이 생길 수 있고
시간도 오래 걸리므로 보류.

### 2.2 conda 로 `clangxx_osx-64` ?

```
$ conda install -y -c conda-forge clangxx_osx-64
CondaToSNonInteractiveError: Terms of Service have not been accepted for the
following channels. Please accept or remove them before proceeding:
    - https://repo.anaconda.com/pkgs/main
    - https://repo.anaconda.com/pkgs/r
```

Anaconda 서버의 **이용약관** 동의가 필요했습니다. 사용자 의사 없이 자동 동의는
부적절하다 판단해 이 경로도 포기했습니다.

### 2.3 그래서 일단 Python 데모만 먼저 푸시

C++ 빌드를 일단 미루고 *지금 즉시 동작하는 Python 데모* 두 개로 학습 자료를 먼저
완성해 GitHub 에 올렸습니다 (커밋 `b840970`).

---

## 3. Xcode Command Line Tools 재설치 — GUI 동의 + 9 분의 다운로드

사용자가 “Xcode CLT 깔고 빌드까지 해 줘” 라고 요청했고, 그래서 다음 명령으로
**시스템 다이얼로그** 를 띄웠습니다.

```
$ xcode-select --install
xcode-select: note: install requested for command line developer tools
```

이 명령은 macOS 의 그래픽 다이얼로그(“Software Update” 대화창) 를 띄우고,
사용자가 거기에서 **Install** 버튼을 직접 눌러야 다운로드가 시작됩니다 — 즉
이 단계는 *반드시 사람의 클릭이 필요* 합니다.

### 3.1 백그라운드 폴링으로 끝을 감지

다운로드/설치는 수 분 ~ 수십 분이 걸릴 수 있어, 다음 셸 루프를 백그라운드로 돌렸습니다.

```bash
for i in $(seq 1 150); do
  if [ -x /Library/Developer/CommandLineTools/usr/bin/clang++ ] \
       && /Library/Developer/CommandLineTools/usr/bin/clang++ --version >/dev/null 2>&1; then
    echo "READY after ${i}0s"
    /Library/Developer/CommandLineTools/usr/bin/clang++ --version
    exit 0
  fi
  sleep 10
done
```

10 초 간격으로 `clang++` 가 실제 동작하는 시점을 감지합니다. 결과:

```
READY after 530s
Apple clang version 14.0.3 (clang-1403.0.22.14.1)
Target: x86_64-apple-darwin22.6.0
Thread model: posix
InstalledDir: /Library/Developer/CommandLineTools/usr/bin
```

**약 9 분** 후 Apple clang 14 가 정상 작동.

---

## 4. 첫 빌드 시도 — Makefile 의 옛 파일명 문제

```
$ cd demo && make clean && make all
make: *** No rule to make target `demo1_cpu_sequential.cpp', needed by
        `../build/demo1_cpu_sequential'.  Stop.
```

이전 단계에서 C++ 소스 파일들의 이름을 `demo1_cpu_*.cpp` → `reference_cpp_*.cpp` 로
바꿨는데, **Makefile 은 옛 이름을 그대로 참조** 하고 있어 빌드가 깨졌습니다.

### 4.1 수정 (Makefile)

```diff
-all: $(BUILD)/demo1_cpu_sequential $(BUILD)/demo2_cpu_parallel
+all: $(BUILD)/reference_cpp_sequential $(BUILD)/reference_cpp_parallel

-$(BUILD)/demo1_cpu_sequential: demo1_cpu_sequential.cpp | $(BUILD)
+$(BUILD)/reference_cpp_sequential: reference_cpp_sequential.cpp | $(BUILD)
        $(CXX) $(CXXFLAGS) $< -o $@

-$(BUILD)/demo2_cpu_parallel: demo2_cpu_parallel.cpp | $(BUILD)
+$(BUILD)/reference_cpp_parallel: reference_cpp_parallel.cpp | $(BUILD)
        $(CXX) $(CXXFLAGS) -pthread $< -o $@
```

### 4.2 두 번째 빌드 — 성공

```
$ make clean && make all
rm -rf ../build
clang++ -O2 -std=c++17 -Wall -Wextra reference_cpp_sequential.cpp -o ../build/reference_cpp_sequential
clang++ -O2 -std=c++17 -Wall -Wextra -pthread reference_cpp_parallel.cpp -o ../build/reference_cpp_parallel
```

경고 없이 깨끗하게 통과.

---

## 5. 첫 측정 — 그리고 즉시 보인 “이상한 점”

`N=1<<20` 으로 5 회씩 돌렸습니다.

```
===== sequential x5 =====
Elapsed = 0.351713 ms   Bandwidth = 35.7761 GB/s
Elapsed = 0.356235 ms   Bandwidth = 35.3219 GB/s
Elapsed = 0.350877 ms   Bandwidth = 35.8613 GB/s
Elapsed = 0.347804 ms   Bandwidth = 36.1782 GB/s
Elapsed = 0.461523 ms   Bandwidth = 27.2639 GB/s

===== parallel default(4t) x5 =====
Elapsed = 2.32915  ms   Bandwidth =  5.40236 GB/s
Elapsed = 1.14532  ms   Bandwidth = 10.9864  GB/s
Elapsed = 1.38106  ms   Bandwidth =  9.11105 GB/s
Elapsed = 0.883764 ms   Bandwidth = 14.2379  GB/s
Elapsed = 1.74531  ms   Bandwidth =  7.20956 GB/s

===== parallel  1t =====   Elapsed = 0.752651 ms
===== parallel  2t =====   Elapsed = 0.805694 ms
===== parallel  8t =====   Elapsed = 3.46157  ms
```

이상한 점이 두 가지였습니다.

1. **순차가 항상 병렬보다 빨랐다.**
   `std::thread` ×4 가 `std::thread` ×1 보다도 안 빨랐고, `×8` 은 노골적으로 느렸음.
2. **순차의 대역폭이 ~36 GB/s** — 이 머신의 DDR4 메모리 이론치(≈25–30 GB/s) 를 넘어
   가는 값이라 “이상하게 빠른” 인상.

→ 가설: **N=1M 은 캐시에 통째로 들어가는 크기** 라 메모리 본체에 안 닿고 있음
(1M × 4B × 2 배열 = 8 MB, i7 의 L3 가 8 MB 안팎). N 을 키우면 진짜 DDR4 한계가 보일 것.

---

## 6. 가설 검증 — N 인자를 추가하고 더 큰 워크로드로 재측정

### 6.1 두 C++ 데모에 CLI 인자 추가

```cpp
// reference_cpp_sequential.cpp
int main(int argc, char **argv) {
    int log2N = 20;
    if (argc >= 2) log2N = std::atoi(argv[1]);
    const int N = 1 << log2N;
    ...
}

// reference_cpp_parallel.cpp
int main(int argc, char **argv) {
    int num_threads = std::thread::hardware_concurrency();
    int log2N = 20;
    if (argc >= 2) num_threads = std::max(1, std::atoi(argv[1]));
    if (argc >= 3) log2N = std::atoi(argv[2]);
    const int N = 1 << log2N;
    ...
}
```

재빌드 후 N = 2²⁰, 2²⁴, 2²⁶ 세 가지로 측정.

### 6.2 측정 결과 (각 모드의 *베스트* 값)

| 모드 | N=2²⁰ (1M) | N=2²⁴ (16M) | N=2²⁶ (64M) |
|---|---:|---:|---:|
| C++ 순차              | **0.348 ms / 35.9 GB/s** | 14.524 ms / 13.9 GB/s | 57.012 ms / 14.1 GB/s |
| C++ 병렬 (`std::thread` ×4) | 0.715 ms / 17.6 GB/s | 15.696 ms / 12.8 GB/s | 85.378 ms / 9.4 GB/s |

가설이 정확히 맞았습니다.

* **N=1M 은 L3 캐시 히트** → 36 GB/s 가 나오는 건 메모리 본체가 아니라 L3 대역폭.
* **N=16M, 64M 부터** 는 캐시를 벗어나 → 13–14 GB/s 라는 진짜 DDR4 한계로 수렴.

### 6.3 “4 스레드가 항상 느린” 이유

| N | 왜 느린가 |
|---|---|
| 1M | 0.35 ms 짜리 작업에 `pthread_create + join` 4번이 이미 더 큼 → launch overhead 지배 |
| 16M | 1 스레드가 이미 DDR4 대역폭에 닿아 있으므로 4 스레드는 *같은 메모리 버스* 를 4 명이 다투기만 함 |
| 64M | 위와 같음 + false sharing / 컨텍스트 스위치 비용 누적 |

다시 말해, 단순 add 는 “계산이 부족해서 느린 것” 이 아니라 “데이터를 읽고 쓰는 메모리
버스가 부족해서 느린 것” 이고, 그 버스를 더 늘릴 방법이 없는 한 *코어를 더 부어도
효과가 없음* 입니다. 이것이 본 머신에서 측정으로 확인된 사실.

---

## 7. 그래서 GPU 가 무엇이 다른가 — 블로그 측정값과의 비교

NVIDIA 블로그의 T4 GPU 최종 결과:

```
Time (%) Time (ns) Instances Category    Operation
   4.4    47,520     1     CUDA_KERNEL  add(int, float *, float *)
```

* T4 의 GDDR6 메모리 대역폭: **320 GB/s** (실측 265 GB/s 까지 활용).
* 본 Mac 의 DDR4 한계: **≈14 GB/s** (실측, N=64M 기준).

비율: **약 19~23×**. 절대 시간은 47 µs vs 350 µs ≈ 7×, 더 큰 N (64M) 으로 가면
더 벌어집니다 (T4 는 캐시 효과가 안 깨지는 워크로드에서 더 강함).

→ “GPU 가 빠른 진짜 이유는 *코어 수 자체* 가 아니라 *코어 수와 메모리 대역폭이 같이*
일정하게 큰 점” 이라는 점이, 우리 Mac 의 부정적인 측정값을 통해 더 분명해졌습니다.
*“CPU 4 코어로 흉내 내려고 해 봤지만 메모리 한계 때문에 안 된다”* — 이걸 직접 본
것이 이 실험의 핵심 수확.

---

## 8. 문서 갱신 + GitHub 푸시

다음 파일을 갱신:

* **README.md** — “Python 데모 결과표” + 새 “C++ 참고 데모 결과표” + “GPU 가 필요한 진짜 이유” 4 항목
* **comparison.md** — 6 열 한 장 요약표(Python / NumPy / MP / C++ 순차 / C++ 병렬 / CUDA),
  새 §4.1.1 “캐시에 들어갈 때만 36 GB/s 가 보인다”, §4.1.2 “4 스레드가 더 느리다 가
  보내는 메시지”
* **demo/Makefile** — 새 파일명 반영
* **demo/reference_cpp_*.cpp** — N(log2) / 스레드 수 CLI 인자

커밋 / 푸시:

```
$ git commit -F /tmp/commit_msg.txt
[main 74ed413] Add measured C++ benchmarks and bandwidth-bound insight
 5 files changed, 109 insertions(+), 50 deletions(-)

$ git push origin main
   b840970..74ed413  main -> main
```

저장소: <https://github.com/leonyoon-3dai/cuda_intro>

---

## 9. 직접 재현하려면

```bash
# (1) 컴파일러 준비 — Mac CLT 가 깨져 있으면
xcode-select --install                       # GUI 다이얼로그에서 Install 클릭

# (2) 빌드
cd cuda_intro/demo
make clean && make all                        # build/reference_cpp_*

# (3) 측정 — 캐시/대역폭 차이 직접 보기
for n in 20 22 24 26; do
  echo "=== N=2^$n ==="
  ../build/reference_cpp_sequential $n
done

# (4) 4 스레드도 같이 측정
for n in 20 22 24 26; do
  ../build/reference_cpp_parallel 4 $n
done

# (5) Python 측 (이미 동작)
python3 demo1_sequential.py
python3 demo2_parallel.py --n 16777216 --procs 4
```

`Bandwidth` 가 **N 을 키우면 한 자릿수 GB/s 영역으로 떨어지는 시점** 이 바로 본 머신의
*L3 캐시가 데이터 두 배열을 더 못 담는 시점* 입니다 (i7 8 MB 안팎). 그 이후로는
스레드를 아무리 늘려도 더 빨라지지 않는 것을 같은 도구로 직접 확인할 수 있습니다.

---

## 10. 한 번 더 정리 — 이 실험이 가르쳐 준 것

1. **컴파일러 환경 정비도 실험의 일부.** 깨진 CLT, 약관에 막힌 conda, GUI 가 필요한
   `xcode-select --install` — 이 모든 단계가 “GPU 코드 한 줄 측정” 보다 시간을 더 많이
   먹기도 합니다.
2. **첫 측정은 보통 *캐시* 를 측정한다.** N=1M / L3 hit → 36 GB/s 는 진짜 메모리가 아닌
   캐시의 모습이라는 점을, N 을 키우는 한 줄 변경으로 노출시켰습니다.
3. **병렬화는 “자유 점심” 이 아님.** memory-bandwidth-bound 한 곳에 코어를 더 부으면
   대부분 손해. 블로그가 GPU 에서 보이는 1932× 의 마법은 *대역폭이 한 자리 큰 디바이스*
   에서나 가능합니다.
4. **결국 GPU 의 역할** — “더 많은 코어 + 그 코어들이 충분히 먹을 수 있는 메모리” 의
   조합. 어느 한 쪽만으로는 같은 결과가 나오지 않으며, 본 Mac 의 측정이 그 점을
   부정형으로 보여 줍니다.
