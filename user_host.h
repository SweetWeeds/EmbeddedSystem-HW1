/** Start of "user_host.h" **/
#ifndef USER_HOST_H
#define USER_HOST_H

#include "user.h"
#include <sys/time.h>
#include <random>

__host__ void host_Concatenate(int *host_mat1, int *host_mat2, int *host_matr, float *p_exec_time);

#endif

/** End of "user_host.h" **/