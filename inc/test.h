#ifndef __TEST_H__
#define __TEST_H__

#include <fstream>
#include "math_constants.h"
#include <gputils.h>
#include <mcutils.h>
#include <assert.h>

/* Struct that contains all the box parameters which must be passed to rayBox*/
typedef struct boxParams boxParams;
struct boxParams {
    double boxMinX, boxMinY, boxMinZ;
    double boxMaxX, boxMaxY, boxMaxZ;
    double orX, orY, orZ;
};

/* Checks for correct intersection (both d2Box and side that is hit) calculated by rayBox()*/
__global__
void ray_box_test(int N, curandState *randS, boxParams *testParams,
                  double *x, double *y, double *z,
                  double *d2Box, int *sideHit);

/* Calls all test functions*/
void run_tests(int N, int nB, int nT, curandState *randS);

#endif