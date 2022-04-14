# MAUI
<b>M</b>onte carlo <b>A</b>lgorithm for <b>U</b>nderstanding photon-tissue <b>I</b>nteraction

## Overview
MAUI is a C++ Cuda program for Monte Carlo simulation of photon propagation. The core functionality of the program is based on the paper by Wang et al. In addition, a number of additional utilities have been added, including: fluorescence, perturbation, and sensitivity matrix generation based on our own heatmap model. 

## Packages used
- CUDA Toolkit
- Numpy
- Matplotlib

## Getting Started
1. Clone the repository: `git clone https://github.com/yyiz/maui.git`
2. `cd maui`
3. Run the make file: `make` (if changes are made, it is recommended you first run: `make clean`)
4. Set the file `maui/inc/constants.h` with the appropriate scene parameters (see file for additional details)
5. `./maui`

## TODO
- Update reconstruction code following modifications (delta abs, new method of sensitivity matrix generation)
- Set up vectorized code for noise / convolution (i.e. be able to apply noise to several tpsf in parallel)
- Set up debug mode - interacting with matlab checker
- Reconstructions with enforcing sparsity in different bases (DCT, etc)
- Closed form solution based on diffusion equation
- Update names: sens->det, hmap -> jacobian

## Completed TODO
- Reflection / refraction
- Modifying simulation to simulate surface scattering, and misc noise (i.e. jitter)
- Simulating absorption, just using delta abs
- Improve scene plotting functions
- Generating transients for absorbed fraction and background signal
- Dealing with fact that lower bound may be beyond voxel bounds
- Handle multiple source and sensor for both heatmap and regular mcml
    - Pass in just source index (position can be calculated with macros)
    - Get back final position of photon, sensor index can be calculated in parent function
- Ray-box intersection in heatmap measurement generation
- Add scene information (timebins, voxel geometry) to output data file
- Test heatmap generation against ground truth
- Determine source of possible error when using multilayered tissue (estimated detected amount lower than expected)
    - Artifact of not checking for roi intersection: if photon is in ROI then scatters into new layer, sLen will change (vs case when new layer is not present)
- Properly handle passing information of which photon hit sensor
- Normalized set of optical and geometric parameters (including source/sensor positions)
- DO NOT PERMANENTLY SET NUM_SENS index to 0

## References
1. L. Wang, L., Jacques, S. L., & Zheng, "MCML - Monte Carlo modeling of light transport in multi-layered tissues," Comput. Methods Programs Biomed., vol. 47, no. 2, pp. 131–146, 1995.
- R. Yao, X. Intes, and Q. Fang, "Direct approach to compute Jacobians for diffuse optical tomography using perturbation Monte Carlo-based photon 'replay,'" Biomed. Opt. Express, vol. 9, no. 10, pp. 4588–4603, Sep. 2018.
- A. Liebert, H. Wabnitz, N. Żołek, and R. Macdonald, "Monte Carlo algorithm for efficient simulation of time-resolved fluorescence in layered turbid media," Opt. Express 16, 13188-13202 (2008)
- A. Williams, S. Barrus, R. Keith, M. P. Shirley, "An efficient and robust ray-box intersection algorithm," Journal of Graphics Tools 10, 54 (2003)

## Additional resources
1. Scudiero, Tony, and Mike Murphy. "Separate Compilation and Linking of Cuda C++ Device Code." _Nvidia Developer Blog_. 22 Apr 2014. https://devblogs.nvidia.com/separate-compilation-linking-cuda-device-code/. Accessed 23 July 2019.
- Harris, Mark. "An Even Easier Introduction to CUDA" _Nvidia Developer Blog_. 25 Jan 2017. https://devblogs.nvidia.com/even-easier-introduction-cuda/. Accessed 23 July 2019.
- talonmies. "What is the canonical way to check for errors using the CUDA runtime API?" _StackOverflow_ 26 Dec 2012. https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api. Accessed 23 July 2019. 
- Michal Hosala. "How to select a GPU with CUDA?" _StackOverflow_ 23 Jan 2015. https://stackoverflow.com/questions/28112485/how-to-select-a-gpu-with-cuda. Accessed 23 July 2019. 
- Van Dam, Andy. _CS1950v: Advanced GPU Programming_. Brown University, 10 April 2011. http://cs.brown.edu/courses/cs195v/lecture/week11.pdf. Accessed 23 July 2019.
- "A Minimal Ray-Tracer: Rendering Simple Shapes (Sphere, Cube, Disk, Plane, etc.)" Scratchapixel 2.0. https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
- "Global Illumination and Path Tracing." Scratchapixel 2.0. https://www.scratchapixel.com/lessons/3d-basic-rendering/global-illumination-path-tracing/global-illumination-path-tracing-practical-implementation. Accessed 12 Apr 2020