#include <jacobian.h>

// Determines histogram / photons to keep for sensitivity matrix based on final position (if hit sensor)
__global__
void sensPhot(int N, double *x, double *y, double *z, 
              double *propTimes,
              int *sensHitInd, int *timeHitInd) {

    // grid stride loop: for phot_i in all photons:
    int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    double invDTime = NUM_BINS / (TIME_MAX - TIME_MIN);
    for (int i = tIdx; i < N; i += stride) { 

        // Initialize index values to zero:
        sensHitInd[i] = 0;
        timeHitInd[i] = 0;

        // Initialize index values
        int sensInd;
        int timeInd;
        double sensX, sensY;
        double xi = x[i]; double yi = y[i]; double zi = z[i]; double propTimeI = propTimes[i];
        bool hitSens = false;

        // Loop over all sensors, test if hit any of them
        for (int sensRow = 0; sensRow < SENS_L; sensRow++) {
            if (hitSens) { // if already hit one sensor, cannot hit another, so break
                break;
            }
            for (int sensCol = 0; sensCol < SENS_W; sensCol++) {
                if (hitSens) { // if already hit one sensor, cannot hit another, so break
                    break;
                }

                // Set position of current sensor for intersection check
                sensInd = sub2ind(sensCol, sensRow, 0, SENS_W, SENS_L);
                sensX = (sensCol * SENS_SEP) + SENS_ORIGX;
                sensY = (sensRow * SENS_SEP) + SENS_ORIGY;

                // if x, y, z coordinates within sensor bounds and within maximum time
                if ((fabs(zi - SENS_ORIGZ) < EPS)
                    && (xi > sensX) && (xi < sensX + SENS_PIXW)
                    && (yi > sensY) && (yi < sensY + SENS_PIXW)
                    && (propTimeI < TIME_MAX)) {

                    timeInd = (int)floor((propTimeI - TIME_MIN) * invDTime);

                    // Set indices needed for sensitivity matrix index in seed replay
                    sensHitInd[i] = sensInd;
                    timeHitInd[i] = timeInd;
                    hitSens = true;
                } else {

                    // Indicate that photon did not hit any sensor by setting index to number of sensors
                    sensHitInd[i] = NUM_SENS;
                }

            }
        }
    }
    return;
}


__global__
void runJacGPU(int N, curandState *randS, double *hmap,
          double *wBack, double *propTimes, 
          int srcRow, int srcCol,
          double *allX, double *allY, double *allZ,
          bool incrHMap=false, int *sensHitInd=NULL, int *timeHitInd=NULL,
          double *pathsX=NULL, double *pathsY=NULL, double *pathsZ=NULL) {

    // Initial starting position based on source position
    int srcInd = sub2ind(srcCol, srcRow, 0, SRC_W, SRC_L);
    double srcX = srcCol * SRC_SEP + SRC_ORIGX;
    double srcY = srcRow * SRC_SEP + SRC_ORIGY;
    double srcZ = SRC_ORIGZ;

    // Initialize optical parameters
    double layerZ[NUMLAYERS + 1] = ALL_LAYERS;
    double layerAbs[NUMLAYERS] = ABS_VEC;
    double layerScat[NUMLAYERS] = SCAT_VEC;
    double layerAnis[NUMLAYERS] = ANIS_VEC;
    double layerRefr[NUMLAYERS] = REFR_VEC;
    double invMuT[NUMLAYERS];
    for (int i = 0; i < NUMLAYERS; i++) {
        invMuT[i] = 1.0 / (layerAbs[i] + layerScat[i]);
    }

    // Initialize misc constants
    double invLightSpeed = 1 / LIGHT_SPEED;
    double invVoxLen = 1 / VOX_SIDELEN;
    double invVoxZlen = 1 / VOX_ZLEN;

    // Evaluate the macros for slab bounds before using later in code
    double minSlabX = SLAB_MINX; 
    double maxSlabX = SLAB_MAXX; 
    double minSlabY = SLAB_MINY; 
    double maxSlabY = SLAB_MAXY; 

    // Declare number of rows and columns as size_t to avoid overflow
    size_t hmapColCnt = (size_t)HMAP_N;

    // Perform grid-stride loop variables
    int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tIdx; i < N; i += stride) {

        // Read per-thread variables from shared array
        curandState randS_i = randS[i];
        propTimes[i] = 0;

        // If incrementing sensitivity matrix, set row of matrix to index, or return if photon misses
        int sensInd;
        int binHit;
        int srcSensTime;
        if (incrHMap) {
            sensInd = sensHitInd[i];
            binHit = timeHitInd[i];
            if (sensInd == NUM_SENS) {
                continue; // Stop current photon if will not contribute to sensitivity matrix
            }
            srcSensTime = sub2ind(binHit, srcInd, sensInd, NUM_BINS, NUM_SRC);
        }

        // Initialize per sample variables
        double x = srcX; double y = srcY; double z = srcZ;
        double uX; double uY; double uZ;
        sampSrc(&uX, &uY, &uZ, &randS_i);
        double propTime = 0.0;
        double sampDist = 0;
        int layerInd = 0; // Indexes arrays for optical and geometrical properties of current layer photon is in
        int k = 0; // index into path array in debug mode

        double packW;
        if (!incrHMap) {
            packW = 1.0;
        } else {
            packW = wBack[i];
        }

        if ((RUNSTATE == RUNDEBUG) && (incrHMap)) {
            int pathInd = N*k + i; 
            pathsX[pathInd] = srcX;
            pathsY[pathInd] = srcY;
            pathsZ[pathInd] = srcZ;
            k++;
        }

        // Subtract surface reflection
        double Rsp = ((1 - layerRefr[layerInd])/(1 + layerRefr[layerInd]));
        Rsp = Rsp * Rsp;

        if (!incrHMap) {
            packW -= Rsp;
        }



        // Main propagation loop
        while (propTime < TIME_MAX) {

            // Determine current voxel coordinate
            int currVoxCol = (int)floor((x-VOX_ORIGX) * invVoxLen);
            int currVoxRow = (int)floor((y-VOX_ORIGY) * invVoxLen);
            int currVoxZ = (int)floor((z-VOX_ORIGZ) * invVoxZlen);

            // Determine optical / geometrical properties based on current layer (layerInd)
            double u_s = layerScat[layerInd];
            double u_a = layerAbs[layerInd];
            double u_t = u_a + u_s;
            double invMuTExt = invMuT[layerInd];
            double g = layerAnis[layerInd];
            double ni = layerRefr[layerInd];
            double d2time = ni * invLightSpeed;
            double minZ = layerZ[layerInd];
            double maxZ = layerZ[layerInd + 1];

            // if sampDist==0: Resample transmission distance
            if (sampDist == 0)
                sampDist = -log(curand_uniform_double(&randS_i));

            int sideHit;
            double propDist;
            int propType = transmit(x, y, z,
                                    uX, uY, uZ,
                                    invVoxLen, invVoxZlen,
                                    minSlabX, minSlabY, minZ,
                                    maxSlabX, maxSlabY, maxZ,
                                    sampDist, invMuTExt,
                                    &propDist, &sideHit,
                                    currVoxCol, currVoxRow, currVoxZ);

            x += propDist * uX;
            y += propDist * uY;
            z += propDist * uZ; 
            propTime += propDist * d2time;
            sampDist -= propDist * u_t;

            if ((RUNSTATE == RUNDEBUG) && (incrHMap) && (k < PLOT_LIM)) {
                int pathInd = N*k + i; 
                pathsX[pathInd] = x;
                pathsY[pathInd] = y;
                pathsZ[pathInd] = z;
                k++;
            }

            // update per-thread sensitivity matrix entry, based on whether a voxel is hit
            if (incrHMap) { 
                int voxI = sub2ind(currVoxCol, currVoxRow, currVoxZ, VOX_W, VOX_L);
                if ((currVoxCol >= 0) && (currVoxRow >= 0) && (currVoxZ >= 0) && 
                    (currVoxCol < VOX_W) && (currVoxRow < VOX_L) && (currVoxZ < VOX_H) &&
                    (voxI < NUM_VOX ) && (voxI >= 0)) {
                    size_t hMapInd = srcSensTime * hmapColCnt + voxI;
                    atomicAdd(&hmap[hMapInd], packW*propDist);
                }
            }

            if (propType == 0) { // regular propagation
                sampDist = 0;
            } else if (propType == 1) { // hit slab bounds
                double nt = 1.0;
                if ((uZ < 0) && (layerInd > 0)) {
                    nt = layerRefr[layerInd - 1];
                } else if ((uZ >= 0) && (layerInd < NUMLAYERS - 1)) {
                    nt = layerRefr[layerInd + 1];
                }

                if (reflRefr(&uX, &uY, &uZ, sideHit, nt, ni, &layerInd, &randS_i) == 0) {
                    // move by small EPS to make sure across boundary (numerical precision)
                    // 2*EPS to counteract EPS that was (possibly) added in transmit()
                    x += 2 * EPS * uX;
                    y += 2 * EPS * uY;
                    z += 2 * EPS * uZ; 
                    continue;
                } else {
                    break;
                }
            } else if (propType == 2) {
                continue;
            }

            // Reduce by absorbed fraction
            if (!incrHMap) {
                packW *= (u_s * invMuTExt);
            }

            // Perform direction changed based on phase function
            hg(g, &randS_i, &uX, &uY, &uZ);

            // Set sampDist to 0, to resample transmission distance
            sampDist = 0;
        }

        // Write back result to global array
        allX[i] = x; allY[i] = y; allZ[i] = z;
        wBack[i] = packW;
        propTimes[i] = propTime;
        randS[i] = randS_i;
    }
    return;
}

// Generate sensitivity matrix
void jacobian(int N, int nB, int nT, int numIters, curandState *randS) {

    // Initialize sensitivity matrix arrays
    double *wBack;
    double *propTimes;
    int *sensHitInd;
    int *timeHitInd;
    double *allX, *allY, *allZ;
    curandState *initRandS;

    double *hmap;

    size_t hmapRowCnt = (size_t)HMAP_M;
    size_t hmapColCnt = (size_t)HMAP_N;
    
    double *pathsX, *pathsY, *pathsZ;
    size_t pathArrSize = (size_t)N*PLOT_LIM;
    if (RUNSTATE == RUNDEBUG) {
        gpu_err_chk(cudaMallocManaged(&pathsX, pathArrSize*sizeof(double)));
        gpu_err_chk(cudaMallocManaged(&pathsY, pathArrSize*sizeof(double)));
        gpu_err_chk(cudaMallocManaged(&pathsZ, pathArrSize*sizeof(double)));

        // Initialize paths
        for (size_t j = 0; j < pathArrSize; j++) {
            pathsX[j] = 0;
            pathsY[j] = 0;
            pathsZ[j] = 0;
        }
    } else {
        pathsX = NULL;
        pathsY = NULL;
        pathsZ = NULL;
    }

    // Allocate memory
    gpu_err_chk(cudaMallocManaged(&wBack, N*sizeof(double)));
    gpu_err_chk(cudaMallocManaged(&propTimes, N*sizeof(double)));
    gpu_err_chk(cudaMallocManaged(&sensHitInd, N*sizeof(int)));
    gpu_err_chk(cudaMallocManaged(&timeHitInd, N*sizeof(int)));
    gpu_err_chk(cudaMallocManaged(&allX, N*sizeof(double)));
    gpu_err_chk(cudaMallocManaged(&allY, N*sizeof(double)));
    gpu_err_chk(cudaMallocManaged(&allZ, N*sizeof(double)));
    gpu_err_chk(cudaMallocManaged(&hmap, hmapRowCnt*hmapColCnt*sizeof(double)));

    // Initialize heatmap
    for (size_t i = 0; i < hmapRowCnt*hmapColCnt; i++) {
        hmap[i] = 0;
    }

    // Initialize array of initial seeds, for seed replay
    gpu_err_chk(cudaMalloc((void**)&initRandS, N*sizeof(curandState)));

    auto runStart = std::chrono::system_clock::now(); // time program runtime

    // Perform mcml over all sources
    for (int srcRow = 0; srcRow < SRC_L; srcRow++) {
        for (int srcCol = 0; srcCol < SRC_W; srcCol++) {
            int srcInd = sub2ind(srcCol, srcRow, 0, SRC_W, SRC_L); // determine 1-d index from src row and col

            // Perform `numIters` iterations, each with N samples
            for (int iterCnt = 0; iterCnt < numIters; iterCnt++) {
                
                if (RUNSTATE == RUNDEBUG) { 
                    runJacGPU <<<nB, nT>>>(N, randS, hmap,
                                      wBack, propTimes,
                                      srcRow, srcCol,
                                      allX, allY, allZ,
                                      true, sensHitInd, timeHitInd,
                                      pathsX, pathsY, pathsZ);
                    gpu_err_chk(cudaDeviceSynchronize()); // block until threads finished

                } else {

                    // Copy initial seeds, to be used for seed replay
                    copy_rand_s <<<nB, nT>>> (N, randS, initRandS);
                    gpu_err_chk(cudaDeviceSynchronize()); // block until threads finished

                    // Perform initial MCML to determine which random-seeds lead to successful path
                    runJacGPU <<<nB, nT>>>(N, randS, NULL,
                                      wBack, propTimes,
                                      srcRow, srcCol,
                                      allX, allY, allZ);
                    gpu_err_chk(cudaDeviceSynchronize()); // block until threads finished

                    // Determine which seeds to keep based on which struck sensor
                    sensPhot <<<nB, nT>>>(N, allX, allY, allZ,
                                          propTimes,
                                          sensHitInd, timeHitInd);
                    gpu_err_chk(cudaDeviceSynchronize()); // block until threads finished

                    // re-run MCML, now only with successful paths, updating sensitivity matrix
                    runJacGPU <<<nB, nT>>>(N, initRandS, hmap,
                                      wBack, propTimes,
                                      srcRow, srcCol,
                                      allX, allY, allZ,
                                      true, sensHitInd, timeHitInd);
                    gpu_err_chk(cudaDeviceSynchronize()); // block until threads finished
                }

                printf("Completed %d Iterations for source: %d/%d\n", iterCnt + 1, srcInd + 1, NUM_SRC);
            }
        }
    }
    // Finish timing program
    auto runEnd = std::chrono::system_clock::now();
    std::chrono::duration<double> runDuration = runEnd - runStart;
    printf("Program runtime: %.17g second(s)\n", runDuration.count());

    // Normalize resulting values
    if (NORMALIZETPSF) {
        for (int i = 0; i < hmapRowCnt*hmapColCnt; i++) {
            hmap[i] /= (N*numIters);
        }
    }
    
    if (RUNSTATE == RUNDEBUG) {
        // Accumulate all 3 arrays
        double *pathsXYZ;
        gpu_err_chk(cudaMallocManaged(&pathsXYZ, 3*pathArrSize*sizeof(double))); // 3 times the length for x, y, z
        for (int p = 0; p < pathArrSize; p++) {
            pathsXYZ[p] = pathsX[p];
        }
        for (int p = 0; p < pathArrSize; p++) {
            pathsXYZ[p + pathArrSize] = pathsY[p];
        }
        for (int p = 0; p < pathArrSize; p++) {
            pathsXYZ[p + 2*pathArrSize] = pathsZ[p];
        }

        writeFile("debug", pathsXYZ, 3*PLOT_LIM, N); 
        gpu_err_chk(cudaFree(pathsX));
        gpu_err_chk(cudaFree(pathsY));
        gpu_err_chk(cudaFree(pathsZ));
        gpu_err_chk(cudaFree(pathsXYZ));
    } else {
        // Using HMAP_M, HMAP_N here is fine because they are not multiplied
        writeFile("hmap", hmap, HMAP_M, HMAP_N); 
    }

    // Free memory
    gpu_err_chk(cudaFree(wBack));
    gpu_err_chk(cudaFree(propTimes));
    gpu_err_chk(cudaFree(sensHitInd));
    gpu_err_chk(cudaFree(timeHitInd));
    gpu_err_chk(cudaFree(allX));
    gpu_err_chk(cudaFree(allY));
    gpu_err_chk(cudaFree(allZ));
    gpu_err_chk(cudaFree(initRandS));
    gpu_err_chk(cudaFree(hmap));
    return;
}
