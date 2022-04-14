// Multilayer background absorption test

// Toggle whether test is run; if true overrides DEBUG or generating measurements
#define RUNTESTS false
#define DEBUG false
#define USE_RAND_SEED true
#define GEN_HMAP false
#define USE_RAY_INTERSEC false

#define NUMGPU 2
#define GPUIDS {4, 5}
#define NUM_SAMPS 10000000
#define NUM_ITERS 10
#define NUM_THREADS 256
#define TPSF_FILENAME "tpsf0.txt"

#define PLOT_NUM_SAMP 100
#define NUM_PLOT_COORDS 3
#define PLOT_LIM 1000 // limit on number of nodes allowed for plotting
#define DEBUG_FILENAME "paths0.csv"

#define HMAP_FILENAME "hmap0.csv"

#define RAND_SEED 2 // USE POSITIVE INTEGER

#define SRC_L 1
#define SRC_W 1
#define SRC_ORIGX 0.0
#define SRC_ORIGY 0.0
#define SRC_ORIGZ 0.0
#define SRC_SEP 0.5

#define SENS_L 1
#define SENS_W 1
#define SENS_ORIGX 2.0
#define SENS_ORIGY 0.0
#define SENS_ORIGZ 0.0
#define SENS_PIXW 0.5
#define SENS_SEP 0.5

#define TIME_MIN 0.0
#define TIME_MAX 100.0
#define NUM_BINS 100

#define VOX_ORIGX (-1.5)
#define VOX_ORIGY (-1.5)
#define VOX_ORIGZ 0
#define VOX_L 8
#define VOX_W 12
#define VOX_H 6
#define VOX_SIDELEN 0.5
#define VOX_ZLEN VOX_SIDELEN

// If in HMap mode, x,y bounds determined by vox bounds
#define SLAB_MINX VOX_ORIGX
#define SLAB_MAXX VOX_ORIGX + VOX_SIDELEN*VOX_W
#define SLAB_MINY VOX_ORIGY
#define SLAB_MAXY VOX_ORIGY + VOX_SIDELEN*VOX_L

// C++ Indexing, ONE MINUS MATLAB INDEX!
#define NUM_ABS 1
#define ABSVOX_ROW {3}
#define ABSVOX_COL {5}
#define ABSVOX_Z {0}
#define ABSVOX_UA 0.05
#define ABSVOX_US 9.0

// Slab Optical properties
#define NUMLAYERS 3
#define ALL_LAYERS {0, 10, 11, 12} // must be length NUMLAYERS+1; includes top and bottom of layers
#define ABS_VEC {0.05, 0.004, 0.02}
#define SCAT_VEC {9.0, 0.009, 10.0}
#define ANIS_VEC {0.9, 0.9, 0.9}
#define REFR_VEC {1.0, 1.0, 1.0}

#define NUM_SENS SENS_L*SENS_W
#define NUM_SRC SRC_L*SRC_W
#define NUM_VOX VOX_L*VOX_W*VOX_H

#define HMAP_M NUM_SRC*NUM_SENS*NUM_BINS
#define HMAP_N NUM_VOX

#define EPS 0.00000001

#define LIGHT_SPEED 0.3 // units: mm/ps 