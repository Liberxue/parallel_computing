#include <stdio.h>

__global__ void twoSum(int* nums, int target, int* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nums[idx];
    for (int i = idx + 1; i < blockDim.x; i++) {
        if (n + nums[i] == target) {
            result[0] = n;
            result[1] = nums[i];
            return;
        }
    }
}

int main() {
    const int size = 5;
    const int target = 9;
    int nums[size] = {2, 7, 11, 15, 3};
    int result[2] = {0};

    int *d_nums, *d_result;
    cudaMalloc((void**)&d_nums, size * sizeof(int));
    cudaMalloc((void**)&d_result, 2 * sizeof(int));
    cudaMemcpy(d_nums, nums, size * sizeof(int), cudaMemcpyHostToDevice);

    twoSum<<<1, size>>>(d_nums, target, d_result);

    cudaMemcpy(result, d_result, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Result: %d, %d\n", result[0], result[1]);

    cudaFree(d_nums);
    cudaFree(d_result);

    return 0;
}
