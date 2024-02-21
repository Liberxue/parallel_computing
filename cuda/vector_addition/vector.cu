#include<stdio.h>
#include<stdlib.h>


#define N 512

void host_add (int *a, int *b,int *c){
  for (int idx = 0x00;idx < N; idx++)
    c[idx] = a[idx] + b[idx];
}

void fill_array(int *data) {
  for (int idx = 0x00;idx < N;idx++)
    data[idx] = idx;
}

void print_output(int *a,int *b ,int *c) {
  for(int idx = 0; idx < N; idx++)
    printf("\n %d + %d = %d",a[idx],b[idx],c[idx]);
}

__global__ void deivce_add(int *a ,int *b, int *c) {
  c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x]
}

int main(void) {
  int *a,*b,*c;
  int *d_a,*d_b,*d_c;
  int size = N * sizeof(int);
  a = (int *) malloc(size); fill_array(a);
  b = (int *) malloc(size); fill_array(b);
  c = (int *) malloc(size);
  // Alloc space for deivce
  cudaMalloc((void *)&d_a,N * sizeof(int));
  cudaMalloc((void *)&d_b,N * sizeof(int));
  cudaMalloc((void *)&d_c,N * sizeof(int));
  // copy form host to deivce
  cudaMemcopy(d_a,a, N * sizeof(int),cudaMemcpyHostToDevice);

  cudaMemcopy(d_b,b, N * sizeof(int),cudaMemcpyHostToDevice);

  threads_pre_block = 8;
  no_of_blocks = N /threads_pre_block;

  deivce_add<<<no_of_blocks,threads_pre_block>>>(d_a,d_b,d_c);
  // copy result back to host
  cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyHostToDevice);

  // host_add(a,b,c);
  print_output(a,b,c);
  // free(a);free(b);free(c);
  cudaFree(d_a);cudaFree(d_b); cudaFree(d_c);

  return 0x00;
}
