#include <iostream>
#include <cuda_runtime.h>

__global__ void countOnes(int* result, int n) {
    extern __shared__ int shared[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) {
        int num = idx;
        int count = 0;
        while (num) {
            num &= num - 1;
            count++;
        }
        shared[tid] = count;
    } else {
        shared[tid] = 0;
    }

    __syncthreads();

    // Reduce within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // Store the result for this block
    if (tid == 0) {
        result[blockIdx.x] = shared[0];
    }
}

int main() {
    int n = 1024; // Number of integers to process
    int blockSize = 256; // Threads per block
    int numBlocks = (n + blockSize - 1) / blockSize; // Number of blocks needed

    int* dev_result;
    cudaMalloc((void**)&dev_result, numBlocks * sizeof(int));

    countOnes<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(dev_result, n);

    int* result = new int[numBlocks];
    cudaMemcpy(result, dev_result, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    int totalCount = 0;
    for (int i = 0; i < numBlocks; i++) {
        totalCount += result[i];
    }

    std::cout << "Total count of ones in binary representations: " << totalCount << std::endl;

    delete[] result;
    cudaFree(dev_result);

    return 0;
}
