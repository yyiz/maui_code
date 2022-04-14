#include <test.h>

/* sideNum key
1: x max
2: x min
3: y max
4: y min
5: z max
6: z min */

__global__
void ray_box_test(int N, curandState *randS, boxParams *testParams,
                  double *x, double *y, double *z,
                  double *d2Box, int *sideHit) {


    double boxMinX = testParams -> boxMinX;
    double boxMinY = testParams -> boxMinY;
    double boxMinZ = testParams -> boxMinZ;
    double boxMaxX = testParams -> boxMaxX;
    double boxMaxY = testParams -> boxMaxY;
    double boxMaxZ = testParams -> boxMaxZ;
    double orX = testParams -> orX;
    double orY = testParams -> orY;
    double orZ = testParams -> orZ;

    int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tIdx; i < N; i += stride) {
        curandState randS_i = randS[i];

        double cosTheta = 2*curand_uniform_double(&randS_i) - 1;;
        double phi = 2 * CUDART_PI * curand_uniform_double(&randS_i);
        double sinTheta = sqrt(1-cosTheta * cosTheta);

        double uX = cos(phi) * sinTheta;
        double uY = sin(phi) * sinTheta;
        double uZ = cosTheta;

        double d2BoxI; int sideHitI;

        ray_box(boxMinX, boxMinY, boxMinZ,
               boxMaxX, boxMaxY, boxMaxZ,
               orX, orY, orZ,
               uX, uY, uZ,
               &d2BoxI, &sideHitI);

        x[i] = orX + d2BoxI*uX;
        y[i] = orY + d2BoxI*uY;
        z[i] = orZ + d2BoxI*uZ;
        d2Box[i] = d2BoxI; sideHit[i] = sideHitI;
    }
    return;
}

void run_tests(int N, int nB, int nT, curandState *randS) {

    double *x; double *y; double *z;
    double *d2Box; int *sideHit;
    boxParams *testParams;
    gpu_err_chk(cudaMallocManaged(&x, N*sizeof(double)));
    gpu_err_chk(cudaMallocManaged(&y, N*sizeof(double)));
    gpu_err_chk(cudaMallocManaged(&z, N*sizeof(double)));
    gpu_err_chk(cudaMallocManaged(&d2Box, N*sizeof(double)));
    gpu_err_chk(cudaMallocManaged(&sideHit, N*sizeof(int)));
    gpu_err_chk(cudaMallocManaged(&testParams, sizeof(boxParams)));

    /* ------------------------------------------------------------------------
    Test 1: Single point in box, with uniform random directions
    Check that correct side is hit
    */
    testParams -> boxMinX = -3;
    testParams -> boxMinY = -4;
    testParams -> boxMinZ = -5;
    testParams -> boxMaxX = -1;
    testParams -> boxMaxY = -2;
    testParams -> boxMaxZ = -3;
    testParams -> orX = -2;
    testParams -> orY = -3;
    testParams -> orZ = -4;

    ray_box_test <<<nB, nT>>>(N, randS, testParams, x, y, z, d2Box, sideHit);
    gpu_err_chk(cudaDeviceSynchronize());

    for (int j = 0; j < N; j++) {
        double xJ = x[j];
        double yJ = y[j];
        double zJ = z[j];
        int sideHitJ = sideHit[j];
        if (sideHitJ == 1)
            assert(fabs(xJ - testParams -> boxMaxX) < EPS);
        else if (sideHitJ == 2)
            assert(fabs(xJ - testParams -> boxMinX) < EPS);
        else if (sideHitJ == 3)
            assert(fabs(yJ - testParams -> boxMaxY) < EPS);
        else if (sideHitJ == 4)
            assert(fabs(yJ - testParams -> boxMinY) < EPS);
        else if (sideHitJ == 5)
            assert(fabs(zJ - testParams -> boxMaxZ) < EPS);
        else if (sideHitJ == 6)
            assert(fabs(zJ - testParams -> boxMinZ) < EPS);
        else
            assert(isinf(d2Box[j]));
    }
    
    gpu_err_chk(cudaFree(x));
    gpu_err_chk(cudaFree(y));
    gpu_err_chk(cudaFree(z));
    gpu_err_chk(cudaFree(d2Box));
    gpu_err_chk(cudaFree(sideHit));
    gpu_err_chk(cudaFree(testParams));
    return;
}