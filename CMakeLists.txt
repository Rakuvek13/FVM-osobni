cmake_minimum_required(VERSION 3.16)
project(FVMComparison LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FVM_USE_MPI "Enable OpenMPI parallelism" OFF)
if(FVM_USE_MPI)
    find_package(MPI REQUIRED CXX)
    add_compile_definitions(FVM_USE_MPI=1)
endif()

add_library(fvm_core
    src/linalg/dense.cpp
    src/linalg/sparse.cpp
    src/mesh/geometry_exact.cpp
    src/mesh/structured_mesh.cpp
    src/mesh/unstructured_mesh.cpp
    src/mesh/mesh_generator.cpp
    src/io/vtk_writer.cpp
    src/solvers/structured_solver.cpp
    src/solvers/unstructured_solver.cpp
)

target_include_directories(fvm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(fvm_core PRIVATE -Wall -Wextra -Wpedantic)
if(FVM_USE_MPI)
    target_link_libraries(fvm_core PUBLIC MPI::MPI_CXX)
endif()

add_executable(problem1 apps/problem1_diagonal_band.cpp)
target_link_libraries(problem1 PRIVATE fvm_core)
target_compile_options(problem1 PRIVATE -Wall -Wextra -Wpedantic)

add_executable(problem2 apps/problem2_unstructured_tube.cpp)
target_link_libraries(problem2 PRIVATE fvm_core)
target_compile_options(problem2 PRIVATE -Wall -Wextra -Wpedantic)

add_executable(problem3 apps/problem3_kpp.cpp)
target_link_libraries(problem3 PRIVATE fvm_core)
target_compile_options(problem3 PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_executable(test_core tests/test_core.cpp)
target_link_libraries(test_core PRIVATE fvm_core)
target_compile_options(test_core PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME test_core COMMAND test_core)

add_executable(test_solvers tests/test_solvers.cpp)
target_link_libraries(test_solvers PRIVATE fvm_core)
target_compile_options(test_solvers PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME test_solvers COMMAND test_solvers)
