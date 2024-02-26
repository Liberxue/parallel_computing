#include <iostream>
#include <cuda_runtime.h>

__device__ int abs(int num) {
    int mask = num >> 31; // Mask is 0xFFFFFFFF for negative numbers, 0x00000000 for non-negative numbers
    return (num + mask) ^ mask;
}

__global__ void absKernel(int* input, int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = abs(input[idx]);
    }
}

int main() {
    int n = 5;
    int input[] = {-5, 10, -15, 20, -25};
    int* dev_input;
    cudaMalloc((void**)&dev_input, n * sizeof(int));
    cudaMemcpy(dev_input, input, n * sizeof(int), cudaMemcpyHostToDevice);

    int* dev_output;
    cudaMalloc((void**)&dev_output, n * sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    absKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_input, dev_output, n);

    int* output = new int[n];
    cudaMemcpy(output, dev_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        std::cout << "Input: " << input[i] << ", Absolute value: " << output[i] << std::endl;
    }

    delete[] output;
    cudaFree(dev_input);
    cudaFree(dev_output);

    return 0;
}
