CXX      := clang++
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra
LDFLAGS  :=

BUILD := ../build

.PHONY: all clean run

all: $(BUILD)/reference_cpp_sequential $(BUILD)/reference_cpp_parallel

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/reference_cpp_sequential: reference_cpp_sequential.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/reference_cpp_parallel: reference_cpp_parallel.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -pthread $< -o $@

run: all
	@echo "=== reference_cpp_sequential ==="
	@$(BUILD)/reference_cpp_sequential
	@echo
	@echo "=== reference_cpp_parallel ==="
	@$(BUILD)/reference_cpp_parallel

clean:
	rm -rf $(BUILD)
