clear; close all; clc;

perturbFilename = "tpsf2.txt";
bkgFilename = "background_tpsf3.txt";
%% pre-loaded measurement file

srcRow = 0;
srcCol = 0;
sensRow = 0;
sensCol = 2;

[headers, nHeaders] = parseHeader(perturbFilename);
allMeasures = readmatrix(perturbFilename, "NumHeaderLines", nHeaders);

SRC_DIM = headers.SRC_DIM;
DET_DIM = headers.SENS_DIM;
NUM_SRC = prod(SRC_DIM);
TIME_MIN = headers.TIME_MIN; TIME_MAX = headers.TIME_MAX;
NBINS = headers.NBINS;
SRC_W = SRC_DIM(2);
DET_W = DET_DIM(2);

srcInd = srcRow*SRC_W + srcCol;
sensInd = sensRow*DET_W + sensCol;

srcSensInd = (sensInd * NUM_SRC) + srcInd + 1;
timeAx = linspace(TIME_MIN, TIME_MAX, NBINS+1);
timeAx = timeAx(1:end-1);
measureI = allMeasures(:,srcSensInd);

plot(timeAx, measureI, 'r', 'LineWidth', 2);
title('TPSF');
xlabel('Time (ps)');

allBkg = readmatrix(bkgFilename, "NumHeaderLines", nHeaders);
bkgI = allBkg(:,srcSensInd);
hold on;
plot(timeAx, bkgI, 'b', 'LineWidth', 2);

legend("Perturbed", "Background");

figure();
plot(timeAx, bkgI - measureI, 'g', 'LineWidth', 2);
title("Difference measurement");
xlabel('Time (ps)');
