CUDA C로 작성한 함수를 Python에서 사용하는 것을 실습합니다.

실습을 위한 cuda 함수는 vector 합 함수입니다.
```
#include <cuda_runtime.h>
#include <stddef.h>

__global__ void vector_add_kernel(float* a, float* b, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

extern "C" void vector_add(float* a, float* b, float* result, int n) {
    float *d_a, *d_b, *d_result;
    size_t size = n * sizeof(float);

    // Allocate device memory
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_result, size);

    // Copy input vectors from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch kernel with enough blocks and threads
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vector_add_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_result, n);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}
```

CMake를 사용하여 빌드한 후 Python에서 사용합니다.

작업 경로로 이동합니다.
```
cd /home/nvidia/workspace/deeplearningwithjetson/Optimization_with_Jetson_Nano/2_CUDA_with_Python
```

빌드합니다.
```
mkdir -p build
cd build
cmake ..
make
```

Python에서 사용합니다.
```
import numpy as np
import ctypes

# C 라이브러리 로드
# 먼저 C 코드를 컴파일하여 shared library (.so 파일)로 만들어야 합니다.
c_lib = ctypes.CDLL('./build/vector_add.so')

# 벡터 합 연산을 수행하는 C 함수 정의
# void vector_add(double* a, double* b, double* result, int n)
c_lib.vector_add.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]

n = 1024
a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)
c = np.zeros_like(a)

# C 함수 호출
c_lib.vector_add(a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                 b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                 c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                 len(a))

# 결과 출력
print(np.max(c -(a+b)))
```