#ifndef __JACOBIAN_H__
#define __JACOBIAN_H__

#include <constants.h>
#include <chrono>
#include <mcutils.h>
#include <iomanip>
#include <curand_kernel.h>

void jacobian(int N, int nB, int nT, int numIters, curandState *randS);

#endif