/** Start of "user_device.h" **/
#ifndef USER_DEVICE_H
#define USER_DEVICE_H

#include "user.h"
#include <cuda_runtime.h>

#define NUM_THREADS_BASE        32
#define NUM_THREADSA_MAX        1024
#define NUM_THREAD_BLKS_FROM    1
#define NUM_THREAD_BLKS_TO      512
#define ITERATION               100

__global__
void device_Concatenate(int *device_mat1, int *device_mat2, int* device_matr, int numOps, int numElements, int mat1_col, int mat2_col, int matr_col);

#endif
/** End of "user_device.h" **/