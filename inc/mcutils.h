#ifndef __MCUTILS_H__
#define __MCUTILS_H__

#include <stdio.h>
#include <fstream>
#include "math_constants.h"
#include <gputils.h>
#include <curand_kernel.h>
#include <constants.h>
#include <chrono>

extern __device__
double plotMat[NUM_PLOT_COORDS * PLOT_NUM_SAMP * PLOT_LIM];

extern __device__
double plotLenVec[NUM_PLOT_COORDS * PLOT_NUM_SAMP];

__device__
void ray_box(double minX, double minY, double minZ,
            double maxX, double maxY, double maxZ,
            double orX, double orY, double orZ,
            double dX, double dY, double dZ,
            double *d2Box, int *sideHit);

__device__
void sampSrc(double *uxPtr, double *uyPtr, double *uzPtr,
             curandState *randSiPtr);

__host__ __device__
inline int sub2ind(int xSub, int ySub, int zSub, 
            int xSize, int ySize) {
    return (zSub * (xSize * ySize)) + (ySub * xSize) + xSub;
}

// Henyey-greenstein phase function
__device__
void inline hg(double g, curandState *randSiPtr, double *dX, double *dY, double *dZ) {
    double uX = *dX; double uY = *dY; double uZ = *dZ;

    double cosTheta;
    if (g == 0) {
        cosTheta = 2*curand_uniform_double(randSiPtr) - 1;
    } else {
        cosTheta = ((1-g*g)/(1-g+2*g*curand_uniform_double(randSiPtr)));
        cosTheta = (1/(2*g)) * ((1 + g*g) - cosTheta*cosTheta);
    }

    double sinTheta = sqrt(1-cosTheta * cosTheta);
    double phi = 2 * CUDART_PI * curand_uniform_double(randSiPtr);
    double cosPhi, sinPhi;
    sincos(phi, &sinPhi, &cosPhi);

    // Calculate scattered direction
    double newDirX;
    double newDirY;
    double newDirZ;
    if (fabs(uZ) > 0.9999) {
        newDirX = sinTheta * cosPhi;
        newDirY = sinTheta * sinPhi;
        if (uZ < 0) {
            newDirZ = -cosTheta;
        } else {
            newDirZ = cosTheta;
        }
    } else {
        double temp = sqrt(1 - uZ*uZ);
        newDirX = sinTheta*(uX*uZ*cosPhi - uY*sinPhi)/temp + uX*cosTheta;
        newDirY = sinTheta*(uY*uZ*cosPhi + uX*sinPhi)/temp + uY*cosTheta;
        newDirZ = -sinTheta*cosPhi*temp+uZ*cosTheta;
    }
    *dX = newDirX;
    *dY = newDirY;
    *dZ = newDirZ;
    return;
}

__device__ 
int inline transmit(double x, double y, double z,
    double uX, double uY, double uZ,
    double invVoxLen, double invVoxZlen,
    double minSlabX, double minSlabY, double minZ,
    double maxSlabX, double maxSlabY, double maxZ,
    double sampDist, double invMuTExt,
    double *propDistPtr, int *sideHitPtr,
    int currVoxCol, int currVoxRow, int currVoxZ) {

    double sLen = sampDist * invMuTExt; // Actual transmission distance in physical units

    // propagate by transmission distance, backtrack if needed
    double tempX = x + (sLen * uX);
    double tempY = y + (sLen * uY);
    double tempZ = z + (sLen * uZ);
    int tempVoxCol = (int)floor((tempX - VOX_ORIGX) * invVoxLen);
    int tempVoxRow = (int)floor((tempY - VOX_ORIGY) * invVoxLen);
    int tempVoxZ = (int)floor((tempZ - VOX_ORIGZ) * invVoxZlen);

    double d2Vox = CUDART_INF;
    double d2SlabBound = CUDART_INF; 
    int voxSide;

    // Determine distance to voxel bounds
    // Do so if: USE_RAY_INTERSEC=true, AND photon escaped current voxel and needs backtrack
    if (USE_RAY_INTERSEC && ((tempVoxCol != currVoxCol) || (tempVoxRow != currVoxRow) || (tempVoxZ != currVoxZ))) {

        double voxMinX = VOX_ORIGX + (currVoxCol * VOX_SIDELEN);
        double voxMinY = VOX_ORIGY + (currVoxRow * VOX_SIDELEN);
        double voxMinZ = VOX_ORIGZ + (currVoxZ * VOX_SIDELEN);
        double voxMaxX = VOX_ORIGX + ((currVoxCol + 1) * VOX_SIDELEN);
        double voxMaxY = VOX_ORIGY + ((currVoxRow + 1) * VOX_SIDELEN); 
        double voxMaxZ = VOX_ORIGZ + ((currVoxZ + 1) * VOX_SIDELEN);
        ray_box(voxMinX, voxMinY, voxMinZ,
                voxMaxX, voxMaxY, voxMaxZ,
                x, y, z,
                uX, uY, uZ,
                &d2Vox, &voxSide);

    }
    // If escaped slab bounds: Determine distance to slab bounds and backtrack
    if ((tempX > maxSlabX) || (tempX < minSlabX)
         || (tempY > maxSlabY) || (tempY < minSlabY)
         || (tempZ < minZ) || (tempZ > maxZ)) {

        ray_box(minSlabX, minSlabY, minZ,
               maxSlabX, maxSlabY, maxZ,
               x, y, z,
               uX, uY, uZ,
               &d2SlabBound, sideHitPtr);
    }

    // If d2SlabBound is shortest OR d2SlabBound == d2Vox, then: propagate by d2SlabBound, update current layerInd
    if ((!isinf(d2SlabBound)) && ((d2SlabBound < d2Vox) || (abs(d2SlabBound - d2Vox) < EPS))) {
        *propDistPtr = d2SlabBound + EPS;
        return 1;
    // Else if d2Vox shortest, propagate by d2Vox
    } else if ((!isinf(d2Vox)) && (d2Vox < d2SlabBound)) {
        *propDistPtr = d2Vox + EPS;
        return 2;
    } 
    // Else sampled distance shortest
    *propDistPtr = sLen;
    return 0;
}

void writeFile(std::string typeWrite, double *matPtr, int numRows, int numCols);

__device__
int reflRefr(double *uxPtr, double *uyPtr, double *uzPtr, 
             int sideHit, double nt, double ni, int *layerIndPtr, curandState *randSiPtr);


// Copy random seeds from src to dest
__global__
void copy_rand_s(int N, curandState *src, curandState *dest);

#endif