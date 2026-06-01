# auris
Auris is a lightweight C/C++ runtime for conversational AI

## Building and Running
```bash
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build -j
./build/auris
```


## Third Party
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — MIT License