#include <iostream>
#include <math.h>
#include <stdio.h>
#include <limits>

#include <test.h>
#include <gputils.h>
#include <jacobian.h>
#include <runmc.h>

/*
Help from the following resources.
-Basic usage of cuda code: https://devblogs.nvidia.com/even-easier-introduction-cuda/
-Sampling random variables: http://cs.brown.edu/courses/cs195v/lecture/week11.pdf
-Cuda error check: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
-GPU Device initialization: https://stackoverflow.com/questions/28112485/how-to-select-a-gpu-with-cuda 
*/

int main(void) {

    init_gpu();

    // Initialize variables
    int N = NUM_SAMPS; // Number of samples   
    int numIters = NUM_ITERS;
    int nT = NUM_THREADS; // number of threads per block, i.e. block size

    if (RUNSTATE == RUNDEBUG) {
        N = PLOT_LIM;
        numIters = 1;
    }

    int nB = (N + nT - 1) / nT; // number of blocks

    curandState *randS;
    unsigned long randSeed;

    // Initialize randstate
    if (USE_RAND_SEED) {
        randSeed = (unsigned long)time(NULL); // Seed curandState with current time
    } else {
        randSeed = (unsigned long)RAND_SEED;
    }

    gpu_err_chk(cudaMalloc((void**)&randS, N*sizeof(curandState)));
    init_rand_s <<<nB, nT>>> (N, randSeed, randS);
    gpu_err_chk(cudaDeviceSynchronize()); // block until threads finished

    printf("Done Initializing curandState\n");

    if (RUNSTATE == RUNTESTS)  {
        printf("TESTING...\n");
        run_tests(N, nB, nT, randS);
        printf("TESTS COMPLETE!\n");
        return 0;
    } else if (RUNSTATE == RUNJAC) {
        printf("GENERATE JACOBIAN\n");
        jacobian(N, nB, nT, numIters, randS);
    } else if (RUNSTATE == RUNSMC) {
        printf("RUN MCML\n");
        run_mcml(N, nB, nT, numIters, randS);
    } else if (RUNSTATE == RUNDEBUG) {
        printf("DEBUGGING\n");
        jacobian(N, nB, nT, numIters, randS);
    } else {
        printf("Unknown run state. Terminating program\n");
    }

    // free memory
    gpu_err_chk(cudaFree(randS));

    return 0;
}