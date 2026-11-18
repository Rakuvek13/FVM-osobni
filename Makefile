CXX ?= clang++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror
USE_MPI ?= 0
NP ?= 4
MPIEXEC ?= mpiexec
ifeq ($(USE_MPI),1)
CXX := mpicxx
CXXFLAGS += -DFVM_USE_MPI=1
BUILD_DIR := build_mpi
else
BUILD_DIR := build
endif

INCLUDES := -Iinclude

CORE_SRCS := \
	src/mesh/geometry_exact.cpp \
	src/mesh/structured_mesh.cpp \
	src/io/vtk_writer.cpp \
	src/solvers/structured_solver.cpp

CORE_OBJS := $(patsubst src/%.cpp,$(BUILD_DIR)/src/%.o,$(CORE_SRCS))
CORE_LIB := $(BUILD_DIR)/libfvm_core.a

TEST_CORE_BIN := $(BUILD_DIR)/test_core
TEST_SOLVERS_BIN := $(BUILD_DIR)/test_solvers

PROB1_BIN := $(BUILD_DIR)/problem1
PROB2_BIN := $(BUILD_DIR)/problem2
PROB3_BIN := $(BUILD_DIR)/problem3

.PHONY: all problems test clean

all: $(CORE_LIB) $(TEST_CORE_BIN) $(TEST_SOLVERS_BIN) $(PROB1_BIN) $(PROB2_BIN) $(PROB3_BIN)

problems: $(PROB1_BIN) $(PROB2_BIN) $(PROB3_BIN)

$(BUILD_DIR)/src/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(CORE_LIB): $(CORE_OBJS)
	@mkdir -p $(dir $@)
	ar rcs $@ $^

$(TEST_CORE_BIN): tests/test_core.cpp $(CORE_LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(CORE_LIB) -o $@

$(TEST_SOLVERS_BIN): tests/test_solvers.cpp $(CORE_LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(CORE_LIB) -o $@

$(PROB1_BIN): apps/problem1_diagonal_band.cpp $(CORE_LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(CORE_LIB) -o $@

$(PROB2_BIN): apps/problem2_smooth_rotation.cpp $(CORE_LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(CORE_LIB) -o $@

$(PROB3_BIN): apps/problem3_kpp.cpp $(CORE_LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(CORE_LIB) -o $@

test: $(TEST_CORE_BIN) $(TEST_SOLVERS_BIN)
ifeq ($(USE_MPI),1)
	$(MPIEXEC) -n $(NP) ./$(TEST_CORE_BIN)
	$(MPIEXEC) -n $(NP) ./$(TEST_SOLVERS_BIN)
else
	./$(TEST_CORE_BIN)
	./$(TEST_SOLVERS_BIN)
endif

clean:
	rm -rf $(BUILD_DIR)
