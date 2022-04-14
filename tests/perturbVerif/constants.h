// constants.h: scene settings for simulation

/* The following set the mode for the simulation to run in.
Toggle different state by setting RUNSTATE to the constant corresponding to the desired run state

RUNTESTS: run maui test cases
RUNJAC: generate sensitivity matrix using photon replay
RUNSMC: generate measurement
RUNDEBUG: generates and stores individual photon trajectories
*/
#define RUNTESTS 0
#define RUNJAC 1
#define RUNSMC 2
#define RUNDEBUG 3
#define RUNSTATE RUNSMC

#define DEBUGPRINT false // Triggers additional prints if errors detected
#define NORMALIZETPSF true // Toggle whether to normalize measurements by total number of photons launched
#define TERMINATEABS true // Toggles whether absorption reduces weight
#define USE_RAY_INTERSEC true // Checks whether scattering site is within voxel boundary, rather than checking for intersection with volume
#define USE_RAND_SEED true // If true, does not seed random number generator
#define RAND_SEED 2  // Integer used for seeded random number generator if USE_RAND_SEED set to false

// if USEPERTURB true: unperturbed tpsf generated with standard monte carlo and stored in background_tpsfX.txt, and perturbed tpsf generated with pMC and stored in tpsfX.txt
// if USEPERTURB false: perturbed tpsf generated with standard monte carlo and stored in background_tpsfX.txt; tpsfX.txt does not have a physically meaningful value
// this flag is largely for debugging purposes; generally it is better to set the flag to true
#define USEPERTURB false

#define GPUID 7 // Select GPU to run on (0-indexed), use nvidia-smi in command line to check available GPUs
#define NUM_SAMPS 10000000 // Number of photons launched in single iteration
#define NUM_ITERS 100 // Number of iterations: total photon samples == NUM_SAMPS*NUM_ITERS
#define NUM_THREADS 256 // Number of Cuda threads to use
#define TPSF_FILENAME "tpsf1.txt" // Filename for measurement output in RUNSMC mode

#define HMAP_FILENAME "hmap0.csv" // Filename for output storing sensitivity matrix

/* Following parameters defined only for debug mode*/
#define PLOT_NUM_SAMP 100 // Number of trajectories to plot
#define NUM_PLOT_COORDS 3 // Constant: number of dimensions of trajectory (usually 3: x,y,z)
#define PLOT_LIM 1000 // Limit on number of nodes allowed for plotting
#define DEBUG_FILENAME "paths0.csv" // Filename for output storing trajectories

// Scene geometry parameters -------------------------------

/* Simulate a SRC_L by SRC_W source array, such that
the top left is positioned at coordinate: (SRC_ORIGX, SRC_ORIGY, SRC_ORIGZ)
and sources are separated by SRC_SEP */
#define SRC_L 1
#define SRC_W 1
#define SRC_ORIGX 7.0
#define SRC_ORIGY 0.0
#define SRC_ORIGZ 0.0
#define SRC_SEP 3.7


/* Simulate a SENS_L by SENS_W sensor array, such that
the top left is positioned at coordinate: (SENS_ORIGX, SENS_ORIGY, SENS_ORIGZ)
and sources are separated by SENS_SEP, and each sensor has sidelength SENS_PIXW */
#define SENS_L 1
#define SENS_W 3
#define SENS_ORIGX (-8.0)
#define SENS_ORIGY (-1.0)
#define SENS_ORIGZ 0.0
#define SENS_PIXW 2.0
#define SENS_SEP 3.7

/* Each transient (collected for a single source-sensor pair) will span TIME_MIN to TIME_MAX
and is discretized into NUM_BINS scalar values */
#define TIME_MIN 0.0
#define TIME_MAX 650.0
#define NUM_BINS 30

/* Defines geometry of a row of the sensitivity matrix (i.e. scene is discretized into voxel grid)
Top left of grid is positioned at coordinate (VOX_ORIGX, VOX_ORIGY, VOX_ORIGZ), along each dimension the grid
consists of VOX_L x VOX_W x VOX_H voxels. Each voxel is a rectangular prism with sidelength: 
VOX_SIDELEN x VOX_SIDELEN x VOX_ZLEN*/
#define VOX_ORIGX -1.0
#define VOX_ORIGY -1.0
#define VOX_ORIGZ 1.0
#define VOX_L 1
#define VOX_W 1
#define VOX_H 1
#define VOX_SIDELEN 2.0
#define VOX_ZLEN 2.0
// ----------------------------------------------

/* The SLAB_ parameters define the limits of the scene (i.e. all photons are terminated if they pass these bounds). 
If in HMap mode, x,y bounds determined by vox bounds */
// #define SLAB_MINX VOX_ORIGX
// #define SLAB_MAXX VOX_ORIGX + VOX_SIDELEN*VOX_W
// #define SLAB_MINY VOX_ORIGY
// #define SLAB_MAXY VOX_ORIGY + VOX_SIDELEN*VOX_L
#define SLAB_MINX (-10.0)
#define SLAB_MAXX 10.0
#define SLAB_MINY (-10.0)
#define SLAB_MAXY 10.0

/* The followign parameters define the absorber image.
RECONTARG defines whether absorber values are defined by the ABSVOX_ parameters (if set to USEMAP)
or if it is read from an image (USEIM).

If RECONTARG set to USEMAP, all absorbers with indices listed by the ABSVOX_ values will be set to absorption coefficient
of ABSVOX_UA, and scattering parameter ABSVOX_US. 

For all 0 <= i < NUM_ABS, voxels with index (ABSVOX_ROW[i], ABSVOX_COL[i], ABSVOX_Z[i] will have optical parameters
defined by ABSVOX_UA (absorption coefficient) and ABSVOX_US (scattering coefficient)*/
#define USEMAP 0
#define USEIM 1
#define RECONTARG USEMAP // IMPORTANT
#define READIM "peppers.csv" // if USEIM, read input file defined by READIM
#define NUM_ABS 1
#define ABSVOX_ROW {0}
#define ABSVOX_COL {0}
#define ABSVOX_Z {0}
#define ABSVOX_UA 1.0
#define ABSVOX_US 9.0

/* The following parameters define the optical and geometrical properties of the slab that is simulated*/ 
#define NUMLAYERS 1 // Number of slab layers
#define ALL_LAYERS {0, 11.0} // Boundaries of each slab layer. Note: slab i is lower and upper bounded by ALL_LAYERS[i], ALL_LAYERS[i+1], respectively
#define ABS_VEC {0} // Absorption coefficients for each layer
#define SCAT_VEC {9.0} // Scattering coefficients for each layer
#define ANIS_VEC {0.9} // Refractive index for each layer
#define REFR_VEC {1.0} // Anisotropic factor for each layer

/* The following precompute the sizes of vectors and matrices used in the linear solver based on the source-detector-voxel grid sizes*/
#define NUM_SENS SENS_L*SENS_W
#define NUM_SRC SRC_L*SRC_W
#define NUM_VOX VOX_L*VOX_W*VOX_H
#define HMAP_M NUM_SRC*NUM_SENS*NUM_BINS
#define HMAP_N NUM_VOX

/*Misc. constants*/
#define EPS 0.00000001 // Used for edge cases involving numerical precision
#define LIGHT_SPEED 0.3 // Speed of light in units of mm/ps, used for converting light propagation distance to time