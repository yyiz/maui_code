#ifndef __RUNMC_H__
#define __RUNMC_H__

#include <constants.h>
#include <chrono>
#include <mcutils.h>
#include <iomanip>
#include <curand_kernel.h>

void run_mcml(int N, int nB, int nT, int numIters, curandState *randS);

void run_debug();

#endif