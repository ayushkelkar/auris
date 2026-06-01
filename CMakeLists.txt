cmake_minimum_required(VERSION 3.20)
project(Auris VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# llama.cpp
set(GGML_AVX2 ON)
set(GGML_AVX_VNNI ON)
set(GGML_CUDA OFF)
set(GGML_METAL OFF)
set(GGML_SYCL ON)
set(GGML_SYCL_F16 ON)
add_subdirectory(third_party/llama.cpp)

# inference module
add_library(inference
    src/inference/inf.cpp
)
target_include_directories(inference PRIVATE
    third_party/llama.cpp/include
    third_party/llama.cpp/ggml/include
)
target_link_libraries(inference PRIVATE llama)

# main executable
add_executable(auris src/main.cpp)
target_include_directories(auris PRIVATE src)
target_link_libraries(auris PRIVATE mmu inference)

# mmu module
add_library(mmu
    src/mmu/vec.cpp
    src/mmu/db.cpp
    src/mmu/search.cpp
)
target_include_directories(mmu PRIVATE src/mmu)
target_compile_options(mmu PRIVATE -mavx2 -mavxvnni)