#include <mcutils.h>

#include <iostream>
#include <iomanip>

__device__
double plotMat[NUM_PLOT_COORDS * PLOT_NUM_SAMP * PLOT_LIM];

__device__
double plotLenVec[NUM_PLOT_COORDS*PLOT_NUM_SAMP];


__device__
void ray_box(double minX, double minY, double minZ,
            double maxX, double maxY, double maxZ,
            double orX, double orY, double orZ,
            double dX, double dY, double dZ,
            double *d2Box, int *sideHit) {

    /* sideNum key
    1: x max
    2: x min
    3: y max
    4: y min
    5: z max
    6: z min */

    double tMin, tMax, tyMin, tyMax, tzMin, tzMax;
    int sideMin, sideMax, tempSideMin, tempSideMax;

    double invDx = 1 / dX;
    double invDy = 1 / dY;
    double invDz = 1 / dZ;

    sideMin = 2;
    sideMax = 1;
    if (dX >= 0) {
        tMin = (minX - orX) * invDx;
        tMax = (maxX - orX) * invDx;
    } else {
        tMin = (maxX - orX) * invDx;
        tMax = (minX - orX) * invDx;
        sideMin = 1;
        sideMax = 2;
    }

    if (dY >= 0) {
        tyMin = (minY - orY) * invDy;
        tyMax = (maxY - orY) * invDy;
        tempSideMin = 4;
        tempSideMax = 3;
    } else {
        tyMin = (maxY - orY) * invDy;
        tyMax = (minY - orY) * invDy;
        tempSideMin = 3;
        tempSideMax = 4;
    }

    if ((tMin > tyMax) || (tyMin > tMax)) {
        *d2Box = CUDART_INF;
        sideHit = 0;
        return;
    }

    if (tyMin > tMin) {
        tMin = tyMin;
        sideMin = tempSideMin;
    }
    if (tyMax < tMax) {
        tMax = tyMax;
        sideMax = tempSideMax;
    }

    if (dZ >= 0) {
        tzMin = (minZ - orZ) * invDz;
        tzMax = (maxZ - orZ) * invDz;
        tempSideMin = 6;
        tempSideMax = 5;
    } else {
        tzMin = (maxZ - orZ) * invDz;
        tzMax = (minZ - orZ) * invDz;
        tempSideMin = 5;
        tempSideMax = 6;
    }

    if ((tMin > tzMax) || (tzMin > tMax)) {
        *d2Box = CUDART_INF;
        sideHit = 0;
        return;
    }

    if (tzMin > tMin) {
        tMin = tzMin;
        sideMin = tempSideMin;
    }
    if (tzMax < tMax) {
        tMax = tzMax;
        sideMax = tempSideMax;
    }

    if (tMax < 0) {
        *d2Box = CUDART_INF; 
        *sideHit = 0;
    } else if (tMin > 0) {
        *d2Box = tMin;
        *sideHit = sideMin;
    } else { // ((tMin < 0) && (tMax > 0))
        *d2Box = tMax;
        *sideHit = sideMax;
    }
    return;
}

/* 
Cosine weighted hemisphere sampling based on instructions from:
https://www.scratchapixel.com/lessons/3d-basic-rendering/global-illumination-path-tracing/global-illumination-path-tracing-practical-implementation
*/
__device__
void sampSrc(double *uxPtr, double *uyPtr, double *uzPtr,
             curandState *randSiPtr) {
    if (SRC_SHAPE == PENCIL) {
        *uxPtr = 0.0;
        *uyPtr = 0.0;
        *uzPtr = 1.0;
    } else if (SRC_SHAPE == COSHEMI) {
        double sampSinT = curand_uniform_double(randSiPtr);
        double sampCosT = sqrt(1-(sampSinT*sampSinT));
        double sampP = 2*CUDART_PI*curand_uniform_double(randSiPtr);
        double cosSampP; 
        double sinSampP;
        sincos(sampP, &sinSampP, &cosSampP);
        *uxPtr = sampSinT*cosSampP;
        *uyPtr = sampSinT*sinSampP;
        *uzPtr = sampCosT;
    } else {
        printf("Invalid source shape!\n");
    }
}

__device__
int reflRefr(double *uxPtr, double *uyPtr, double *uzPtr, 
             int sideHit, double nt, double ni, int *layerIndPtr, curandState *randSiPtr) {

    double uX = *uxPtr;
    double uY = *uyPtr;
    double uZ = *uzPtr;
    int layerInd = *layerIndPtr;
    double cosSurfNorm = abs(uZ);
    double ai = acos(cosSurfNorm);
    double at = asin((ni / nt) * sin(ai));

    if (sideHit < 5) {
        return 1;
    }

    bool isRefl;
    if ((ni > nt) && (ai > asin(nt / ni))) {
        isRefl = true;
    } else {
        double R;
        if (cosSurfNorm > (1 - 0.000001)) { // if angle very close to normal
            R = ((ni - nt) / (ni + nt)) * ((ni - nt) / (ni + nt));
        } else {
            double sinAiMinAt = sin(ai-at);
            double sinAiPlusAt = sin(ai+at);
            double tanAiMinAt = tan(ai-at);
            double tanAiPlusAt = tan(ai+at);
            R = (0.5)*(((sinAiMinAt*sinAiMinAt)/(sinAiPlusAt*sinAiPlusAt)) + 
                ((tanAiMinAt*tanAiMinAt)/(tanAiPlusAt*tanAiPlusAt)));
        }
        if (curand_uniform_double(randSiPtr) <= R) {
            isRefl = true;
        } else {
            isRefl = false;
        }
    }

    if (isRefl) {
        uZ *= -1;
    } else {

        uX *= (ni / nt);
        uY *= (ni / nt);
        double newDirZ = cos(at);
        if (uZ < 0) {
            newDirZ *= -1;
        }
        uZ = newDirZ;

        if (sideHit == 5) {
            layerInd += 1;
        } else { // layerInd == 6
            layerInd -= 1;
        }

        if ((layerInd < 0) || (layerInd >= NUMLAYERS)) {
            *uxPtr = uX;
            *uyPtr = uY;
            *uzPtr = uZ;
            *layerIndPtr = layerInd;
            return 1;
        }

    }
    *layerIndPtr = layerInd;
    *uxPtr = uX;
    *uyPtr = uY;
    *uzPtr = uZ;
    return 0;
}


void writeFile(std::string typeWrite, double *matPtr, int numRows, int numCols) {
    std::ofstream fp; 
    std::string saveFilename("data/");
    std::setprecision(std::numeric_limits< double >::max_digits10);

    if (typeWrite == "tpsf") {
        saveFilename += TPSF_FILENAME;
    } else if (typeWrite == "hmap") {
        saveFilename += HMAP_FILENAME;
    } else if (typeWrite == "tpsf_background") {
        std::string backgroundStr("background_");
        saveFilename += backgroundStr + TPSF_FILENAME;
    } else if (typeWrite == "gnd_truth") {
        std::string backgroundStr("truth_");
        saveFilename += backgroundStr + TPSF_FILENAME;
    } else if (typeWrite == "debug") {
        saveFilename += DEBUG_FILENAME;
    } else {
        return;
    }
    fp.open(saveFilename);

    fp << "NUM_SAMPS=" << NUM_SAMPS << std::endl;
    fp << "NUM_ITERS=" << NUM_ITERS << std::endl;
    fp << "SRC_DIM=" << SRC_L << "," << SRC_W << std::endl;
    fp << "SRC_ORIG=" << SRC_ORIGX << "," << SRC_ORIGY << "," << SRC_ORIGZ << std::endl;
    fp << "SRC_SEP=" << SRC_SEP << std::endl;
    fp << "SENS_DIM=" << SENS_L << "," << SENS_W << std::endl;
    fp << "SENS_ORIG=" << SENS_ORIGX << "," << SENS_ORIGY << "," << SENS_ORIGZ << std::endl;
    fp << "SENS_PIXW=" << SENS_PIXW << std::endl;
    fp << "SENS_SEP=" << SENS_SEP << std::endl;
    fp << "VOX_DIM=" << VOX_L << "," << VOX_W << "," << VOX_H << std::endl;
    fp << "VOX_ORIG=" << VOX_ORIGX << "," << VOX_ORIGY << "," << VOX_ORIGZ << std::endl;
    fp << "VOX_SIDELEN=" << VOX_SIDELEN << std::endl;
    fp << "VOX_ZLEN=" << VOX_ZLEN << std::endl;
    fp << "TIME_MIN=" << TIME_MIN << std::endl;
    fp << "TIME_MAX=" << TIME_MAX << std::endl;
    fp << "NUM_BINS=" << NUM_BINS << std::endl;
    fp << "RECONTARG=" << RECONTARG << std::endl;
    fp << "NORMALIZETPSF=" << NORMALIZETPSF << std::endl;
    fp << "TERMINATEABS=" << TERMINATEABS << std::endl;
    fp << "USE_RAY_INTERSEC=" << USE_RAY_INTERSEC << std::endl;
    fp << "USE_RAND_SEED=" << USE_RAND_SEED << std::endl;
    fp << "RAND_SEED=" << RAND_SEED << std::endl;
    fp << "GPUID=" << GPUID << std::endl;
    fp << "RECONTARG=" << RECONTARG << std::endl;
    fp << "READIM=" << READIM << std::endl;
    fp << "NUMLAYERS=" << NUMLAYERS << std::endl;
    double layerZ[NUMLAYERS + 1] = ALL_LAYERS;
    double layerAbs[NUMLAYERS] = ABS_VEC;
    double layerScat[NUMLAYERS] = SCAT_VEC;
    double layerAnis[NUMLAYERS] = ANIS_VEC;
    double layerRefr[NUMLAYERS] = REFR_VEC;
    for (int i = 0; i < NUMLAYERS + 1; i++) {
        if (i == 0) {
            fp << "ALL_LAYERS={" << layerZ[0];
        }
        else {
            fp << "," << layerZ[i];
        }
        if (i == NUMLAYERS) {
            fp << "}" << std::endl;
        }
    }
    for (int i = 0; i < NUMLAYERS; i++) {
        if (i == 0) {
            fp << "ABS_VEC={" << layerAbs[0];
        }
        else {
            fp << "," << layerAbs[i];
        }
        if (i == NUMLAYERS-1) {
            fp << "}" << std::endl;
        }
    }
    for (int i = 0; i < NUMLAYERS; i++) {
        if (i == 0) {
            fp << "SCAT_VEC={" << layerScat[0];
        }
        else {
            fp << "," << layerScat[i];
        }
        if (i == NUMLAYERS-1) {
            fp << "}" << std::endl;
        }
    }
    for (int i = 0; i < NUMLAYERS; i++) {
        if (i == 0) {
            fp << "ANIS_VEC={" << layerAnis[0];
        }
        else {
            fp << "," << layerAnis[i];
        }
        if (i == NUMLAYERS-1) {
            fp << "}" << std::endl;
        }
    }
    for (int i = 0; i < NUMLAYERS; i++) {
        if (i == 0) {
            fp << "REFR_VEC={" << layerRefr[0];
        }
        else {
            fp << "," << layerRefr[i];
        }
        if (i == NUMLAYERS-1) {
            fp << "}" << std::endl;
        }
    }
    if ((typeWrite == "tpsf") || (typeWrite == "tpsf_background")) {
        fp << "ABSVOX_UA=" << ABSVOX_UA << std::endl;
        fp << "NUM_ABS=" << NUM_ABS << std::endl;
        double absRowArr[NUM_ABS] = ABSVOX_ROW; 
        double absColArr[NUM_ABS] = ABSVOX_COL;
        double absZArr[NUM_ABS] = ABSVOX_Z;
        fp << "ABSVOX_ROW=";
        for (int absI = 0; absI < NUM_ABS; absI++) {
            if (absI == NUM_ABS-1) {
                fp << absRowArr[absI] << std::endl;
            } else {
                fp << absRowArr[absI] << ",";
            }
        }
        fp << "ABSVOX_COL=";
        for (int absI = 0; absI < NUM_ABS; absI++) {
            if (absI == NUM_ABS-1) {
                fp << absColArr[absI] << std::endl;
            } else {
                fp << absColArr[absI] << ",";
            }
        }
        fp << "ABSVOX_Z=";
        for (int absI = 0; absI < NUM_ABS; absI++) {
            if (absI == NUM_ABS-1) {
                fp << absZArr[absI] << std::endl;
            } else {
                fp << absZArr[absI] << ",";
            }
        }
    }

    // Write heatmap matrix
    for (size_t row = 0; row < numRows; row++) {
        for (size_t col = 0; col < numCols; col++) {
            // int matInd = sub2ind(col, row, 0, numCols, numRows);
            size_t colSize = (size_t)numCols;
            size_t matInd = (row * colSize) + col;
            if (col != numCols - 1) {
                fp << matPtr[matInd] << ", ";
            } else {
                fp << matPtr[matInd];
            }
        }
        fp << std::endl;
    }


    std::cout << "Generated file: " << saveFilename << std::endl;
    fp.close();
}

// Copy random seeds from src to dest
__global__
void copy_rand_s(int N, curandState *src, curandState *dest) {
    // Initialize grid-stride loop variables
    int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tIdx; i < N; i += stride) { 
        dest[i] = src[i];
    }
}

