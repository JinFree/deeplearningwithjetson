CUDA C로 작성한 함수를 Python에서 사용하는 것을 실습합니다.

실습 과정에서 vector 합 함수를 구현하고 각각의 환경에서 속도를 비교해보겠습니다.

우선 Python 구현은 아래와 같습니다.
```
def vector_add_python(a, b, c, n):
    for i in range(n):
        c[i] = a[i] + b[i]
    return c

def vector_add_numpy(a, b, c, n):
    c = a + b
    return c
```

CUDA 구현을 확인해보겠습니다.
```
#include <cuda_runtime.h>
#include <stddef.h>

__global__ void vector_add_kernel(float* a, float* b, float* result, int n) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = thread ; idx < n ; idx += stride) {
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
    int numBlocks = 4;
    vector_add_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_result, n);

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
cd /home/nvidia/workspace/deeplearningwithjetson/Optimization_with_Jetson_Nano/2_CUDA_with_python
```

빌드합니다.
```
mkdir -p build
cd build
cmake ..
make
```

Python에서 시간을 확인합니다.
```
python3 /home/nvidia/workspace/deeplearningwithjetson/Optimization_with_Jetson_Nano/2_CUDA_with_python/vector_add.py
```

아래와 유사한 시간을 확인할 수 있습니다.
```
vector_add_python 2회 반복 평균: 19475.573603ms
vector_add_numpy 10회 반복 평균: 109.621709ms
vector_add 10회 반복 평균: 241.101823ms
```