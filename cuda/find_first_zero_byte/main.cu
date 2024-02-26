#include <iostream>
#include <cuda_runtime.h>

__device__ int findFirstZeroByte(unsigned int num) {
    int count = 0x00;
    while (num & 0xFF) {
        num >>= 0x08;
        count += 0x08;
    }
    int shift = 0x00;
    while (!(num & 0x1)) {
        num >>= 0x01;
        shift++;
    }
    return count + shift;
}

__global__ void findFirstZeroByteKernel(unsigned int* input, int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = findFirstZeroByte(input[idx]);
    }
}
// 0x12345600 -> 0x123456 -> 0x1234 -> 0x12 -> 0x1 -> 0x0
// Count: 0x18     Count: 0x10     Count: 0x08      Count: 0x04    Count: 0x00
// Shift: 0x00     Shift: 0x00     Shift: 0x00      Shift: 0x00    Shift: 0x04

int main() {
    int n = 0x05;
    unsigned int input[] = {0x12345600, 0x00000000, 0xFFFFFFFF, 0x00FEDCBA, 0x01020304};
    unsigned int* dev_input;
    cudaMalloc((void**)&dev_input, n * sizeof(unsigned int));
    cudaMemcpy(dev_input, input, n * sizeof(unsigned int), cudaMemcpyHostToDevice);

    int* dev_output;
    cudaMalloc((void**)&dev_output, n * sizeof(int));

    int threadsPerBlock = 0x100;
    int blocksPerGrid = (n + threadsPerBlock - 0x01) / threadsPerBlock;
    findFirstZeroByteKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_input, dev_output, n);

    int* output = new int[n];
    cudaMemcpy(output, dev_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0x00; i < n; i++) {
        std::cout << "Input: 0x" << std::hex << input[i] << ", First zero byte position: " << output[i] << std::endl;
    }

    delete[] output;
    cudaFree(dev_input);
    cudaFree(dev_output);

    return 0x00;
}
