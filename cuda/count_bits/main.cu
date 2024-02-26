#include <iostream>
#include <cuda_runtime.h>

__device__ int countBits(unsigned int num) {
    int count = 0;
    while (num) {
        count += num & 1;
        num >>= 1;
    }
    return count;
}

__global__ void countBitsKernel(unsigned int* input, int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = countBits(input[idx]);
    }
}

int main() {
    int n = 5;
    unsigned int input[] = {0x80000000, 0x12345678, 0x00000001, 0xFEDCBA98, 0x01020304};
    unsigned int* dev_input;
    cudaMalloc((void**)&dev_input, n * sizeof(unsigned int));
    cudaMemcpy(dev_input, input, n * sizeof(unsigned int), cudaMemcpyHostToDevice);

    int* dev_output;
    cudaMalloc((void**)&dev_output, n * sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    countBitsKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_input, dev_output, n);

    int* output = new int[n];
    cudaMemcpy(output, dev_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        std::cout << "Input: 0x" << std::hex << input[i] << ", Number of bits: " << output[i] << std::endl;
    }

    delete[] output;
    cudaFree(dev_input);
    cudaFree(dev_output);

    return 0;
}
