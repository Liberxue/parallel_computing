#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

__global__ void threeSumKernel(int* nums, int n, int* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    for (int i = idx + 1; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (nums[idx] + nums[i] + nums[j] == 0) {
                atomicAdd(result, 1);
            }
        }
    }
}

int threeSum(std::vector<int>& nums) {
    int n = nums.size();
    int result = 0;

    int* dev_nums;
    cudaMalloc((void**)&dev_nums, n * sizeof(int));
    cudaMemcpy(dev_nums, nums.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    int* dev_result;
    cudaMalloc((void**)&dev_result, sizeof(int));
    cudaMemset(dev_result, 0, sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    threeSumKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_nums, n, dev_result);

    cudaMemcpy(&result, dev_result, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_nums);
    cudaFree(dev_result);

    return result;
}

int main() {
    std::vector<int> nums = {-1, 0, 1, 2, -1, -4};
    int result = threeSum(nums);
    std::cout << "Number of unique triplets that sum up to zero: " << result << std::endl;

    return 0;
}
