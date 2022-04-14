#ifndef __GPUTILS_H__
#define __GPUTILS_H__

#include <stdio.h>
#include <curand_kernel.h>
#include <constants.h>

#define gpu_err_chk(ans) { gpu_assert((ans), __FILE__, __LINE__); }

/* Inline error check for GPU functions */
void gpu_assert(cudaError_t code, const char *file, int line);

/* GPU initialization function
Initialization 2nd GPU if available, otherwise uses 1st (prints warning in this case)
Checks that device major number is up to date (>= 2)*/
int init_gpu();

/*Initialize curandState on multiple GPU threads
Needed for RNG on GPU */
__global__
void init_rand_s(int N, unsigned long seed, curandState *randS);

#endif