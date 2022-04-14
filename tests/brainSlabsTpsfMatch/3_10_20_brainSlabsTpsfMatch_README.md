git commit: e9a833071c502d57e0a5f4435573a49ad14bd088

Testing on standard 3-layer brain scene. This was compared to forward model used by Hielscher group at Columbia. Good match with DE. 

Matlab code:

srcSideLen = 1;
srcOriginX = 10 - srcSideLen/2; srcOriginY = 20 - srcSideLen/2; srcOriginZ = 0;
sensOriginY = 20 - sensPixW/2; sensOriginZ = 0;
sensXArr = [14 18 22 26 30] - sensPixW/2;
maxTime = 1600;
nBins = 80; % This can be downsampled
layer1 = 0;
layer2 = 10; %Tianjin University of Science and Technology in China - 6.5mm for man 7.1 for women
layer3 = 12;
layer4 = 30;
numLayers = 3; % 3 layers, 4 separation bands
minX = 0; minY = 0; minZ = 0;
maxX = 40; maxY = 40; maxZ = 30;
u_a1 = 0.05; u_s1 = 9; g1 = 0.9; n1 = 1.4;
u_a2 = 0.004; u_s2 = 0.009; g2 = 0.9; n2 = 1.4;
u_a3 = 0.02; u_s3 = 10; g3 = 0.9; n3 = 1.4;


Output log:

Completed 100 Iterations for source: 1/1
Absorbed Average 0: 3.0523440639962566e-05
Background average 0: 2.1822602670398605e-07
Direct surface reflection: 0.027777777777777766
Program runtime: 2163.4803673179999 second(s)
Generated file: data/truth_tpsf0.txt
Generated file: data/tpsf0.txt
Generated file: data/background_tpsf0.txt
