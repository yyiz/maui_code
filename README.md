# MAUI
<b>M</b>onte carlo <b>A</b>lgorithm for <b>U</b>nderstanding photon-tissue <b>I</b>nteraction

## Overview
MAUI is a Cuda/C++ program for Monte Carlo simulation of photon propagation. The core functionality of the program is based on the paper by Wang et al [1]. In addition, a number of additional utilities have been added. This includes generation of the Jacobian (sensitivity) matrix [2], perturbation monte carlo, and fluorescence simulation [3]. 

## Packages used
- CUDA Toolkit (tested on v10.2)
- Matlab (tested on R2019b)

## Getting Started
1. Clone the repository: `git clone https://github.com/yyiz/maui.git`
2. `cd maui`
3. `mkdir data obj`

## How to run?
### Running the Simulator
0. The first step is to set the desired parameters in a header file in the `settings/` directory. The template file is available at `settings/TEMPLATE.h`. This file controls all relevant parameters of the simulation, including the GPUID, source-detector configuration, time-of-flight settings, etc.
1. To verify the correctness of your settings, a visualization script is provided: `plotConfig.h`. Within this file, change the `configFname` variable to the desired configuration header file. The output plot will show the source-detector configuration and slab geometry.
2. There are two options to run the simulator: compiling from the runfile or running the matlab launch script:
   - The recommended method is to use the `mat/launchMaui.m` file. This script will launch the number of processes set by `nProcs`. Each of these processes will use the parameters set in the header file specified by `configFname`, EXCEPT that the parameters specified by the keys of the dictionary `params` will be reset to the values associated with the key. Note that the value of each key should be an array of length `nProcs`. This will set the i-th process to the value of the i-th array element. Finally, make sure to set the `mauiDir` variable to the maui path on your current machine.
   - Alternatively, you can call the run script: `./run.sh run filename.h` where `filename.h` is the name of the desired header file to run. In this case, only one process will be launched using the exact parameters set by `filename.h`. 
3. Wait for the program to terminate. If the simulator is launched with the matlab script, the process is launched in the background and the progress can be tracked by reading the `data/log_*` file. If the simulator is launched with the run script, the process will be started in the foreground. Therefore, when using the run script, it is recommended to launch the simulator with `tmux`. For either simulator launch method, the process can also be tracked by calling `nvidia-smi`, which will display a process(es) named `maui` until the simulation(s) have terminated.
4. After the processes have completed, check that the raw data files have been saved in the `data/` directory. Standard monte carlo (SMC) simulations will create multiple files named `*tpsf*.txt` while jacobian simulations will create `hmap*.csv` files.

### Cleaning up Raw Data

1. After running the simulator, move the raw data files into a folder `target_dir/tempfiles`, where `target_dir` can be replaced with any name.
2. In `mc2mat.m`, change `defaultPath` and `basedir` variables to the path to the `target_dir` and the `target_dir` name, respectively. 
3. In `mc2mat.m`, set the `saveJ` and/or `saveM` variables to true if the output files were for standard monte carlo and/or jacobian simulations, respectively. 
4. Since `launchMaui.m` launches multiple processes and results in multiple raw data files, the output files should each have an associated number. Set the `jFnum` and `mFnum` variables to arrays of integers corresponding to the integers in the raw data filename. Again, `jFnum` and `mFnum` refer to the jacobian and standard monte carlo simulations, respectively. 
5. This will create `.mat` file(s) containing the sensitivity matrix and/or simulated time-domain measurements for jacobian and standard monte carlo simulations, respectively. These `.mat` files also include the simulation parameters. The output names are set by the `jDestname` and `mDestname` for jacobian and standard monte carlo simulations, respectively.

### Misc. Notes
A batch file `Makefile.bat` is included for launching the simulator on an Nvidia-GPU enabled Windows machine. This script has not been thoroughly tested for compatibility. Therefore, Nvidia-GPU enabled linux machines are highly recommended.

## References
1. L. Wang, L., Jacques, S. L., & Zheng, "MCML - Monte Carlo modeling of light transport in multi-layered tissues," Comput. Methods Programs Biomed., vol. 47, no. 2, pp. 131–146, 1995.
2. R. Yao, X. Intes, and Q. Fang, "Direct approach to compute Jacobians for diffuse optical tomography using perturbation Monte Carlo-based photon 'replay,'" Biomed. Opt. Express, vol. 9, no. 10, pp. 4588–4603, Sep. 2018.
3. A. Liebert, H. Wabnitz, N. Żołek, and R. Macdonald, "Monte Carlo algorithm for efficient simulation of time-resolved fluorescence in layered turbid media," Opt. Express 16, 13188-13202 (2008)
4. A. Williams, S. Barrus, R. Keith, M. P. Shirley, "An efficient and robust ray-box intersection algorithm," Journal of Graphics Tools 10, 54 (2003)

## Additional resources
1. Scudiero, Tony, and Mike Murphy. "Separate Compilation and Linking of Cuda C++ Device Code." _Nvidia Developer Blog_. 22 Apr 2014. https://devblogs.nvidia.com/separate-compilation-linking-cuda-device-code/. Accessed 23 July 2019.
2. Harris, Mark. "An Even Easier Introduction to CUDA" _Nvidia Developer Blog_. 25 Jan 2017. https://devblogs.nvidia.com/even-easier-introduction-cuda/. Accessed 23 July 2019.
3. Crovella, Robert. "What is the canonical way to check for errors using the CUDA runtime API?" _StackOverflow_ 26 Dec 2012. https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api. Accessed 23 July 2019. 
4. Michal Hosala. "How to select a GPU with CUDA?" _StackOverflow_ 23 Jan 2015. https://stackoverflow.com/questions/28112485/how-to-select-a-gpu-with-cuda. Accessed 23 July 2019. 
5. Van Dam, Andy. _CS1950v: Advanced GPU Programming_. Brown University, 10 April 2011. http://cs.brown.edu/courses/cs195v/lecture/week11.pdf. Accessed 23 July 2019.
6. "A Minimal Ray-Tracer: Rendering Simple Shapes (Sphere, Cube, Disk, Plane, etc.)" Scratchapixel 2.0. https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
7. "Global Illumination and Path Tracing." Scratchapixel 2.0. https://www.scratchapixel.com/lessons/3d-basic-rendering/global-illumination-path-tracing/global-illumination-path-tracing-practical-implementation. Accessed 12 Apr 2020
