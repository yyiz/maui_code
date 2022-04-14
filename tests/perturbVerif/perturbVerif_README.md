Verification on perturbation monte carlo. 

git commit: 1e8006d8e6f03d80de650a8e73452980873eb356

files stored in: /mnt/data0/yz142/3_22_20_perturb

Remarks:
- The runtimes for run 1 and run 2 are slow primarily because they were launched on the same GPUs
- Run 1 and run 2 were run with: `#define USE_RAY_INTERSEC false`, which is an approximation to perturbation monte carlo, assuming that full transmission distance is based on optical properties at previous perturbation site

----------------------------------------------------
Run 1 (tpsf2): generated with perturbation monte carlo. TPSF estimated by multiplying the weight of each detected sample by a perturbation weight; Does NOT change optical properties in the middle of transmission distance if moves in/out of perturbation.
#define USEPERTURB true
#define USE_RAY_INTERSEC false


Output:

Completed 100 Iterations for source: 1/1
Absorbed Average 0: 0.00014093965341402365
Background average 0: 0.00017751
Absorbed Average 1: 0.00044256228877923975
Background average 1: 0.00067272100000000004
Absorbed Average 2: 0.0020210085996914882
Background average 2: 0.0029810129999999998
Direct surface reflection: 0
Program runtime: 1265.2763404889999 second(s)
Generated file: data/truth_tpsf2.txt
Generated file: data/tpsf2.txt
Generated file: data/background_tpsf2.txt


----------------------------------------------------
Run 2 (background_tpsf3): generated with monte carlo. If photon sample in perturbation, modify absorption coefficient if in perturbation; Does NOT change optical properties in the middle of transmission distance if moves in/out of perturbation.
#define USEPERTURB false
#define USE_RAY_INTERSEC false

Output:

Completed 100 Iterations for source: 1/1
Absorbed Average 0: 0.00014156569203587971
Background average 0: 0.00014156569203587971
Absorbed Average 1: 0.00044306853972598365
Background average 1: 0.00044306853972598365
Absorbed Average 2: 0.0020213317487164503
Background average 2: 0.0020213317487164503
Direct surface reflection: 0
Program runtime: 1213.654414117 second(s)
Generated file: data/truth_tpsf3.txt
Generated file: data/tpsf3.txt
Generated file: data/background_tpsf3.txt


-----------------------------------------------------
Run 3 (tpsf0): generated with perturbation monte carlo. TPSF estimated by multiplying the weight of each detected sample by a perturbation weight; DOES change optical properties in the middle of transmission distance if moves in/out of perturbation.
#define USEPERTURB true
#define USE_RAY_INTERSEC true 


Output:

Completed 100 Iterations for source: 1/1
Absorbed Average 0: 0.00014033709776641642
Background average 0: 0.00017648800000000001
Absorbed Average 1: 0.00044278955286656028
Background average 1: 0.00067411200000000004
Absorbed Average 2: 0.0020182343315766969
Background average 2: 0.002981437
Direct surface reflection: 0
Program runtime: 3057.6833023240001 second(s)
Generated file: data/truth_tpsf0.txt
Generated file: data/tpsf0.txt
Generated file: data/background_tpsf0.txt







--------------------------------------------------
Run 4 (background_tpsf1): generated with monte carlo. If photon sample in perturbation, modify absorption coefficient if in perturbation; DOES change optical properties in the middle of transmission distance if moves in/out of perturbation.
#define USEPERTURB false
#define USE_RAY_INTERSEC true 

Output:

Absorbed Average 0: 0.0001407552507856061
Background average 0: 0.0001407552507856061
Absorbed Average 1: 0.00044284407650420381
Background average 1: 0.00044284407650420381
Absorbed Average 2: 0.0020159592792747746
Background average 2: 0.0020159592792747746
Direct surface reflection: 0
Program runtime: 3358.5236496819998 second(s)
Generated file: data/truth_tpsf1.txt
Generated file: data/tpsf1.txt
Generated file: data/background_tpsf1.txt

----------------------------------------------------
Run 5 (tpsf4)

#define USE_RAY_INTERSEC false 
#define USEPERTURB true

#define NUM_ABS 8
#define ABSVOX_ROW {0, 0, 0, 0, 1, 1, 1, 1}
#define ABSVOX_COL {0, 0, 1, 1, 0, 0, 1, 1}
#define ABSVOX_Z {0, 1, 0, 1, 0, 1, 0, 1}
#define ABSVOX_UA 1.0
#define ABSVOX_US 9.0



Output:

Completed 100 Iterations for source: 1/1
Absorbed Average 0: 0.00014085958402147003
Background average 0: 0.00017693900000000001
Absorbed Average 1: 0.00044321537815010862
Background average 1: 0.00067422599999999999
Absorbed Average 2: 0.0020212592814360139
Background average 2: 0.0029809020000000001
Direct surface reflection: 0
Program runtime: 593.83664253799998 second(s)
Generated file: data/truth_tpsf4.txt
Generated file: data/tpsf4.txt
Generated file: data/background_tpsf4.txt


----------------------------------------------------
Run 6 (tpsf5)

#define USE_RAY_INTERSEC false 
#define USEPERTURB true

#define NUM_ABS 8
#define ABSVOX_ROW {0, 0, 0, 0, 1, 1, 1, 1}
#define ABSVOX_COL {0, 0, 1, 1, 0, 0, 1, 1}
#define ABSVOX_Z {0, 1, 0, 1, 0, 1, 0, 1}
#define ABSVOX_UA 1.0
#define ABSVOX_US 9.0

#define NUMLAYERS 2 // Number of slab layers
#define ALL_LAYERS {0, 1.0, 11.0} // Boundaries of each slab layer. Note: slab i is lower and upper bounded by ALL_LAYERS[i], ALL_LAYERS[i+1], respectively
#define ABS_VEC {0, 0} // Absorption coefficients for each layer
#define SCAT_VEC {9.0, 9.0} // Scattering coefficients for each layer
#define ANIS_VEC {0.9, 0.9} // Refractive index for each layer
#define REFR_VEC {1.0, 1.0} // Anisotropic factor for each layer

Output:

Completed 100 Iterations for source: 1/1
Absorbed Average 0: 0.00014093574510201917
Background average 0: 0.00017741799999999999
Absorbed Average 1: 0.00044226976735854001
Background average 1: 0.00067294299999999996
Absorbed Average 2: 0.0020497163677201634
Background average 2: 0.0029809189999999998
Direct surface reflection: 0
Program runtime: 651.25724339099997 second(s)
Generated file: data/truth_tpsf5.txt
Generated file: data/tpsf5.txt
Generated file: data/background_tpsf5.txt

