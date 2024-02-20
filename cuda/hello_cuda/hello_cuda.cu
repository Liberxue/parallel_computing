#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DS_time.h"
#include <stdio.h>

__global__ void helloCUDA(void)
{
    printf("Hello CUDA Form GPU !\n");
}

int main_helloCUDA(void)
{
    printf("Helllo GPU From CPU !\n");
    helloCUDA <<<1, 10>>>();
    return 0;
}