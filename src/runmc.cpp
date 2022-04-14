#include <runmc.h>

__global__
void detect_phot(int N, double *x, double *y, double *z, 
              double *wOut, double *wBack, double *propTimes, int *sensHitInds) {

    // grid stride loop: for phot_i in all photons:
    int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tIdx; i < N; i += stride) { 

        double sensX, sensY;
        double xi = x[i]; double yi = y[i]; double zi = z[i]; double propTimeI = propTimes[i];
        sensHitInds[i] = NUM_SENS;
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
                int sensInd = sub2ind(sensCol, sensRow, 0, SENS_W, SENS_L);
                sensX = (sensCol * SENS_SEP) + SENS_ORIGX;
                sensY = (sensRow * SENS_SEP) + SENS_ORIGY;

                // if x, y, z coordinates within sensor bounds and within maximum time
                if ((fabs(zi - SENS_ORIGZ) < EPS)
                    && (xi > sensX) && (xi < sensX + SENS_PIXW)
                    && (yi > sensY) && (yi < sensY + SENS_PIXW)
                    && (propTimeI < TIME_MAX) && (propTimeI > TIME_MIN)) {

                    hitSens = true;
                    sensHitInds[i] = sensInd;
                }  
            }
        }
    }
    return;
}

__global__
void mcgpu(int N, curandState *randS, double *hmap, double *absMap,
          double *wOut, double *wBack, double *propTimes, 
          int srcRow, int srcCol,
          double *allX, double *allY, double *allZ) {

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

    // Perform grid-stride loop variables
    int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tIdx; i < N; i += stride) {

        // Read per-thread variables from shared array
        curandState randS_i = randS[i];
        wOut[i] = 0;
        wBack[i] = 0;
        propTimes[i] = 0;
        double perturbLen = 0;

        // Initialize per sample variables
        double x = srcX; double y = srcY; double z = srcZ;
        double uX; double uY; double uZ;
        sampSrc(&uX, &uY, &uZ, &randS_i);
        double packW = 1.0;
        double propTime = 0.0;
        double sampDist = 0;
        int layerInd = 0; // Indexes arrays for optical and geometrical properties of current layer photon is in

        // Subtract surface reflection
        double Rsp = ((1 - layerRefr[layerInd])/(1 + layerRefr[layerInd]));
        Rsp = Rsp * Rsp;
        if (TERMINATEABS) {
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

            double diffAbs = 0;
            if ((currVoxRow >= 0) && (currVoxRow < VOX_L) &&
                (currVoxCol >= 0) && (currVoxCol < VOX_W) &&
                (currVoxZ >= 0) && (currVoxZ < VOX_H)) {

                int voxInd = sub2ind(currVoxCol, currVoxRow, currVoxZ, VOX_W, VOX_L);
                diffAbs = absMap[voxInd];
            }  

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

            if (diffAbs != 0) {
                // Perturbation length needs to weighted by absorption: (mua_bkg - u_a)*len
                perturbLen += diffAbs * propDist;
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
                    x += 2 * EPS * uX; // move by small EPS to make sure across boundary (numerical prec)
                    y += 2 * EPS * uY;
                    z += 2 * EPS * uZ; 
                    continue;
                } else {
                    break;
                }
            } else if (propType == 2) { // Else if d2Vox shortest, propagate by d2Vox
                continue;
            }

            // Reduce by absorbed fraction
            packW *= (u_s * invMuTExt);
            
            // Perform direction changed based on phase function
            hg(g, &randS_i, &uX, &uY, &uZ);

            // Set sampDist to 0, to resample transmission distance
            sampDist = 0;
        }

        // Write back result to global array
        allX[i] = x; allY[i] = y; allZ[i] = z;
        wOut[i] = packW * exp(-perturbLen);
        wBack[i] = packW;
        propTimes[i] = propTime;
        randS[i] = randS_i;
    }
    return;
}

__global__
void bin_phots(int N, double *wOut, double *wBack, double *propTimes,
               int srcInd, int *sensHitInds,
               double *accumPhotBins, double *accumPhotBack,
               double *avgVec, double *backAvg) {

    double invDTime = NUM_BINS / (TIME_MAX - TIME_MIN);

    // Perform grid-stride loop variables
    int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tIdx; i < N; i += stride) {
        int sensInd = sensHitInds[i];
        if (sensInd == NUM_SENS) {
            continue;
        }

        double propTimeI = propTimes[i];
        int timeInd = (int)floor((propTimeI - TIME_MIN) * invDTime);
        int srcSensI = sub2ind(srcInd, sensInd, 0, NUM_SRC, NUM_SENS);

        int srcSensTimeI = sub2ind(srcSensI, timeInd, 0, NUM_SRC*NUM_SENS, NUM_BINS);

        double binI = wOut[i];
        double binBackI = wBack[i];

        // read histogram for both absorbed and background photons
        atomicAdd(&accumPhotBins[srcSensTimeI], binI);
        atomicAdd(&accumPhotBack[srcSensTimeI], binBackI);

        // calculate integrated number of photons detected
        atomicAdd(&avgVec[srcSensI], binI);
        atomicAdd(&backAvg[srcSensI], binBackI);

    }
    return;
}

void run_mcml(int N, int nB, int nT, int numIters, curandState *randS) {

    // Initialize cuda arrays
    double *wOut, *wBack;
    double *propTimes;
    double *allX, *allY, *allZ;
    int *sensHitInds;
    double *avgVec, *backAvg, *accumPhotBins, *accumPhotBack;
    double *absMap;

    // Allocate memory
    gpu_err_chk(cudaMallocManaged(&wOut, N*sizeof(double)));
    gpu_err_chk(cudaMallocManaged(&wBack, N*sizeof(double)));
    gpu_err_chk(cudaMallocManaged(&propTimes, N*sizeof(double)));
    gpu_err_chk(cudaMallocManaged(&sensHitInds, N*sizeof(int)));
    gpu_err_chk(cudaMallocManaged(&allX, N*sizeof(double)));
    gpu_err_chk(cudaMallocManaged(&allY, N*sizeof(double)));
    gpu_err_chk(cudaMallocManaged(&allZ, N*sizeof(double)));

    gpu_err_chk(cudaMallocManaged(&avgVec, NUM_SRC*NUM_SENS*sizeof(double)));
    gpu_err_chk(cudaMallocManaged(&backAvg, NUM_SRC*NUM_SENS*sizeof(double)));
    gpu_err_chk(cudaMallocManaged(&accumPhotBins, NUM_BINS*NUM_SRC*NUM_SENS*sizeof(double)));
    gpu_err_chk(cudaMallocManaged(&accumPhotBack, NUM_BINS*NUM_SRC*NUM_SENS*sizeof(double)));
    gpu_err_chk(cudaMallocManaged(&absMap, NUM_VOX*sizeof(double)));

    for (int i = 0; i < NUM_VOX; i++) {
        absMap[i] = 0;
    }

    if (RECONTARG == USEIM) {
        FILE *absFP;
        std::string openFilename("scenes/");
        openFilename = openFilename + READIM;
        const char *openFilenamePtr = openFilename.c_str();
        absFP = fopen(openFilenamePtr, "r");
        for (int row = 0; row < VOX_L; row++) {
            for (int col = 0; col < VOX_W; col++) {
                int voxInd = sub2ind(col, row, 0, VOX_W, VOX_L);
                if (!fscanf(absFP, "%lf", &absMap[voxInd])) {
                    printf("Failed to read all data!\n");
                    fclose(absFP);
                    return;
                }
            }
        }

    } else if (RECONTARG == USEMAP) {
        int absRowArr[NUM_ABS] = ABSVOX_ROW; 
        int absColArr[NUM_ABS] = ABSVOX_COL;
        int absZArr[NUM_ABS] = ABSVOX_Z;
        for (int i = 0; i < NUM_ABS; i++) {
            int absRow = absRowArr[i];
            int absCol = absColArr[i];
            int absZ = absZArr[i];
            int voxInd = sub2ind(absCol, absRow, absZ, VOX_W, VOX_L);
            absMap[voxInd] = ABSVOX_UA;
        }
    } else {
        printf("Invalid absorber map, check input\n");
        return;
    }

    for (int i = 0; i < NUM_BINS*NUM_SRC*NUM_SENS; i++) {
        accumPhotBins[i] = 0;
        accumPhotBack[i] = 0;
    }

    // Initialize array values to 0 for CPU arrays (for GPU arrays, this is done in device code)
    for (int initI = 0; initI < NUM_SRC*NUM_SENS; initI++) {
        avgVec[initI] = 0;
        backAvg[initI] = 0;
    }
    auto runStart = std::chrono::system_clock::now(); // start timing

    // Perform mcml over all sources
    for (int srcRow = 0; srcRow < SRC_L; srcRow++) {
        for (int srcCol = 0; srcCol < SRC_W; srcCol++) {
            int srcInd = sub2ind(srcCol, srcRow, 0, SRC_W, SRC_L); // determine 1-d index from src row and col

            // Perform `numIters` iterations, each with N samples
            for (int iterCnt = 0; iterCnt < numIters; iterCnt++) {

                // Perform core functionality: MCML on GPU
                mcgpu <<<nB, nT>>>(N, randS, NULL, absMap,
                                  wOut, wBack, propTimes,
                                  srcRow, srcCol,
                                  allX, allY, allZ);
                gpu_err_chk(cudaDeviceSynchronize()); // block until threads finished

                // Accumulate and bin results from MCML
                detect_phot <<<nB, nT>>>(N, allX, allY, allZ,
                                      wOut, wBack, propTimes, sensHitInds);
                gpu_err_chk(cudaDeviceSynchronize()); // block until threads finished

                bin_phots <<<nB, nT>>>(N, wOut, wBack, propTimes,
                                       srcInd, sensHitInds,
                                       accumPhotBins, accumPhotBack,
                                       avgVec, backAvg);
                gpu_err_chk(cudaDeviceSynchronize()); // block until threads finished

                printf("Completed %d Iterations for source: %d/%d\n", iterCnt + 1, srcInd + 1, NUM_SRC);
                
            }
        }
    }

    // Finish timing
    auto runEnd = std::chrono::system_clock::now();
    std::chrono::duration<double> runDuration = runEnd - runStart;

    // Normalize histograms
    if (NORMALIZETPSF) {
        double normalization = (double)(numIters * N);
        for (int srcSensI = 0; srcSensI < NUM_SRC*NUM_SENS; srcSensI++) {
            avgVec[srcSensI] /= normalization;
            backAvg[srcSensI] /= normalization;
            for (int timeI = 0; timeI < NUM_BINS; timeI++) {
                int srcSensTimeI = sub2ind(srcSensI, timeI, 0, NUM_SRC*NUM_SENS, NUM_BINS);
                accumPhotBins[srcSensTimeI] /= normalization;
                accumPhotBack[srcSensTimeI] /= normalization;
            }
        }
    }

    // Calculate reflected light for printing
    double layerRefr[NUMLAYERS] = REFR_VEC;
    int topLayerInd = 0;
    double Rsp = ((1 - layerRefr[topLayerInd])/(1 + layerRefr[topLayerInd]));
    Rsp = Rsp * Rsp;

    // Print integrated photon counts
    for (int srcSensIndPrint = 0; srcSensIndPrint < NUM_SENS*NUM_SRC; srcSensIndPrint++) {
        printf("Absorbed Average %d: %.17g\n", srcSensIndPrint, avgVec[srcSensIndPrint]);
        printf("Background average %d: %.17g\n", srcSensIndPrint, backAvg[srcSensIndPrint]);
    }
    printf("Direct surface reflection: %0.17g\n", Rsp);

    // Print program runtime
    printf("Program runtime: %.17g second(s)\n", runDuration.count());

    // Write results to memory
    writeFile("gnd_truth", absMap, VOX_L, VOX_W);
    writeFile("tpsf", accumPhotBins, NUM_BINS, NUM_SRC*NUM_SENS);
    writeFile("tpsf_background", accumPhotBack, NUM_BINS, NUM_SRC*NUM_SENS);

    // Free memory
    gpu_err_chk(cudaFree(wOut));
    gpu_err_chk(cudaFree(wBack));
    gpu_err_chk(cudaFree(propTimes));
    gpu_err_chk(cudaFree(sensHitInds));
    gpu_err_chk(cudaFree(allX));
    gpu_err_chk(cudaFree(allY));
    gpu_err_chk(cudaFree(allZ));


    gpu_err_chk(cudaFree(avgVec));
    gpu_err_chk(cudaFree(backAvg));
    gpu_err_chk(cudaFree(accumPhotBins));
    gpu_err_chk(cudaFree(accumPhotBack));
    gpu_err_chk(cudaFree(absMap));
}

void run_debug() {
    return;
}
