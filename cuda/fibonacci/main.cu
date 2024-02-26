#include <iostream>
#include <cuda_runtime.h>

__global__ void fibonacci(int n, int* result) {
    if (blockIdx.x == 0) {
        if (threadIdx.x == 0) {
            result[0] = 0;
            result[1] = 1;
        }
    }

    __syncthreads();

    int a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        int temp = a + b;
        a = b;
        b = temp;
    }

    if (threadIdx.x == 0) {
        result[n] = b;
    }
}

int main() {
    int n = 10; // count fibonacci
    int* dev_result;
    cudaMalloc((void**)&dev_result, (n + 1) * sizeof(int));

    fibonacci<<<1, 1>>>(n, dev_result);

    int result;
    cudaMemcpy(&result, dev_result + n, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_result);

    std::cout << "Fibonacci number at position " << n << ": " << result << std::endl;

    return 0;
}
