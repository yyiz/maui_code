#include <gputils.h>

void gpu_assert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "gpu_assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}


int init_gpu() {
    // Initialize GPU
    int numGPU;
    cudaDeviceProp devProp;
    gpu_err_chk(cudaGetDeviceCount(&numGPU));
    if (numGPU > GPUID) {
        gpu_err_chk(cudaSetDevice(GPUID));
    }
    else {
        int zeroInd = 0;
        printf("Warning: Cannot assign desired GPU. Assigning to GPU: %d\n", zeroInd);
        gpu_err_chk(cudaSetDevice(GPUID));
    }
    gpu_err_chk(cudaGetDeviceProperties(&devProp, GPUID));
    if (devProp.major >= 2) {
        return 0;
    }
    return 1;
}


__global__
void init_rand_s(int N, unsigned long seed, curandState *randS) {
    int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tIdx; i < N; i += stride) {
        curand_init(seed, i, 0, &randS[i]);
    }
}

