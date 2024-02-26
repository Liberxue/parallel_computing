#include <iostream>
#include <cuda_runtime.h>

__global__ void bitManipulation(int* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Bitwise AND operation
        int resultAnd = input[idx] & 0xF; // Take the lower four bits of input[idx]
        // Bitwise OR operation
        int resultOr = input[idx] | 0xF0; // Set the lower four bits of input[idx] to 1
        // Bitwise XOR operation
        int resultXor = input[idx] ^ 0xFF; // Invert each bit of input[idx]
        // Left shift operation
        int resultLeftShift = input[idx] << 1; // Shift input[idx] binary representation one bit to the left
        // Right shift operation
        int resultRightShift = input[idx] >> 1; // Shift input[idx] binary representation one bit to the right

        printf("Input: %d, AND: %d, OR: %d, XOR: %d, Left Shift: %d, Right Shift: %d\n",
               input[idx], resultAnd, resultOr, resultXor, resultLeftShift, resultRightShift);
    }
}

int main() {
    int n = 5;
    int input[] = {1, 2, 3, 4, 5};
    int* dev_input;
    cudaMalloc((void**)&dev_input, n * sizeof(int));
    cudaMemcpy(dev_input, input, n * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    bitManipulation<<<blocksPerGrid, threadsPerBlock>>>(dev_input, n);

    cudaFree(dev_input);

    return 0;
}
