#include <iostream>
#include <cuda_runtime.h>

// Swap odd and even bits in a 32-bit int
// Odd bits:   101010...
// Even bits:  010101...
// To swap, extract odd bits using 0xaaaaaaaa, shift right by 1,
// extract even bits using 0x55555555, shift left by 1, and OR the results
__device__ unsigned int swapOddEvenBits(unsigned int num) {
    // Extract odd bits, shift right by 1;
    // Extract even bits, shift left by 1, then combine
    return ((num & 0xaaaaaaaa) >> 0x01) | ((num & 0x55555555) << 0x01);
}

__global__ void swapOddEvenBitsKernel(unsigned int* input, unsigned int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = swapOddEvenBits(input[idx]);
    }
}

// input

//   31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
//   -------------------------------------------------------------------------------------------------
//   |     1     |     0     |     1     |     0     |     1     |     0     |     1     |     0     |
//   -------------------------------------------------------------------------------------------------

// output

//   31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
//   -------------------------------------------------------------------------------------------------
//   |     0     |     1     |     0     |     1     |     0     |     1     |     0     |     1     |
//   -------------------------------------------------------------------------------------------------

// Input:  0xAABBCCDD
// Binary: 1010 1010 1011 1011 1100 1100 1101 1101
//
// Mask1:  0xAAAAAAAA (1010 1010 1010 1010 1010 1010 1010 1010)
// Mask2:  0x55555555 (0101 0101 0101 0101 0101 0101 0101 0101)
//
// Step 1: (Input & Mask1) >> 1
// Result: 0101 0101 0101 0101 1010 1010 1010 1010
//
// Step 2: (Input & Mask2) << 1
// Result: 1010 1010 1010 1010 0101 0101 0101 0101
//
// Output: 0x5555AAAA (0101 0101 0101 0101 1010 1010 1010 1010)


int main() {
    int n = 0x05;
    unsigned int input[] = {0xAAAAAAAA, 0x55555555, 0x12345678, 0xFEDCBA98, 0x01020304};
    unsigned int* dev_input;
    cudaMalloc((void**)&dev_input, n * sizeof(unsigned int));
    cudaMemcpy(dev_input, input, n * sizeof(unsigned int), cudaMemcpyHostToDevice);

    unsigned int* dev_output;
    cudaMalloc((void**)&dev_output, n * sizeof(unsigned int));

    int threadsPerBlock = 0x100;
    int blocksPerGrid = (n + threadsPerBlock - 0x01) / threadsPerBlock;
    swapOddEvenBitsKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_input, dev_output, n);

    unsigned int* output = new unsigned int[n];
    cudaMemcpy(output, dev_output, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    for (int i = 0x00; i < n; i++) {
        std::cout << "Input: 0x" << std::hex << input[i] << ", Output: 0x" << output[i] << std::endl;
    }

    delete[] output;
    cudaFree(dev_input);
    cudaFree(dev_output);

    return 0x00;
}
