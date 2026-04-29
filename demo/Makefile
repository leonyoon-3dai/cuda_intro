CXX      := clang++
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra
LDFLAGS  :=

BUILD := ../build

.PHONY: all clean run

all: $(BUILD)/demo1_cpu_sequential $(BUILD)/demo2_cpu_parallel

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/demo1_cpu_sequential: demo1_cpu_sequential.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/demo2_cpu_parallel: demo2_cpu_parallel.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -pthread $< -o $@

run: all
	@echo "=== demo1 (sequential) ==="
	@$(BUILD)/demo1_cpu_sequential
	@echo
	@echo "=== demo2 (parallel) ==="
	@$(BUILD)/demo2_cpu_parallel

clean:
	rm -rf $(BUILD)
