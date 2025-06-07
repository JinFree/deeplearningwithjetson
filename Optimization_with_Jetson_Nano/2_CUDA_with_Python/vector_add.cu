#include <cuda_runtime.h>
#include <stddef.h>
#include <cstring>


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
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}