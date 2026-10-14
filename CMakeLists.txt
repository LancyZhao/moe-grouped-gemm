cmake_minimum_required(VERSION 3.18)
project(moe_grouped_gemm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# BUILD_CUDA=OFF builds only the CPU reference + its test, so correctness can be
# verified (and CI-checked) on a machine with no GPU/nvcc.
option(BUILD_CUDA "Build the CUDA benchmark executable" ON)

add_library(ggref src/reference.cpp)
target_include_directories(ggref PUBLIC include)

enable_testing()
add_executable(test_reference tests/test_reference.cpp)
target_link_libraries(test_reference PRIVATE ggref)
add_test(NAME reference COMMAND test_reference)

if(BUILD_CUDA)
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 89)
  endif()
  find_package(CUDAToolkit REQUIRED)
  add_executable(bench src/main.cu src/kernels.cu)
  target_include_directories(bench PRIVATE include)
  target_link_libraries(bench PRIVATE ggref CUDA::cublas)
endif()
