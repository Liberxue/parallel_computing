#include <math.h>
#include "cuda_runtime.h"
#include "kernel.h"

// declare the kernel function
__global__ void kernel_sum(const float* A, const float* B, float* C, int n_el);

// function which invokes the kernel
void sum(const float* A, const float* B, float* C, int n_el) {

  // declare the number of blocks per grid and the number of threads per block
  int threadsPerBlock,blocksPerGrid;

  // use 1 to 512 threads per block
  if (n_el<512){
    threadsPerBlock = n_el;
    blocksPerGrid   = 1;
  } else {
    threadsPerBlock = 512;
    blocksPerGrid   = ceil(double(n_el)/double(threadsPerBlock));
  }

  // invoke the kernel
  kernel_sum<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, n_el);
}

// kernel
__global__ void kernel_sum(const float* A, const float* B, float* C, int n_el)
{
  // calculate the unique thread index
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  // perform tid-th elements addition
  if (tid < n_el) C[tid] = A[tid] + B[tid];
}
