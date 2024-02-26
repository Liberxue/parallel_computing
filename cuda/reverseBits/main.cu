#include <iostream>
#include <cuda_runtime.h>

__device__ unsigned int reverseBits(unsigned int num) {
    num = (num >> 1) & 0x55555555 | (num << 1) & 0xaaaaaaaa;
    num = (num >> 2) & 0x33333333 | (num << 2) & 0xcccccccc;
    num = (num >> 4) & 0x0f0f0f0f | (num << 4) & 0xf0f0f0f0;
    num = (num >> 8) & 0x00ff00ff | (num << 8) & 0xff00ff00;
    num = (num >> 16) & 0x0000ffff | (num << 16) & 0xffff0000;
    return num;
}

__global__ void reverseBitsKernel(unsigned int* input, unsigned int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = reverseBits(input[idx]);
    }
}

int main() {
    int n = 5;
    unsigned int input[] = {0xAAAAAAAA, 0x55555555, 0x12345678, 0xFEDCBA98, 0x01020304};
    unsigned int* dev_input;
    cudaMalloc((void**)&dev_input, n * sizeof(unsigned int));
    cudaMemcpy(dev_input, input, n * sizeof(unsigned int), cudaMemcpyHostToDevice);

    unsigned int* dev_output;
    cudaMalloc((void**)&dev_output, n * sizeof(unsigned int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    reverseBitsKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_input, dev_output, n);

    unsigned int* output = new unsigned int[n];
    cudaMemcpy(output, dev_output, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        std::cout << "Input: 0x" << std::hex << input[i] << ", Output: 0x" << output[i] << std::endl;
    }

    delete[] output;
    cudaFree(dev_input);
    cudaFree(dev_output);

    return 0;
}
