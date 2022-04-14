clear; close all; clc;

addpath(genpath("../../lib/"));

nMeasFiles = 200;

datdir = "9_13_21_neural_representation_training";
basedir = "/mnt/data0/yz142";
datpath = sprintf("%s/%s/tempfiles", basedir, datdir);

for k = 1:nMeasFiles
    bkgPath = sprintf("%s/background_tpsf%d.txt", datpath, k-1);
    measPath = sprintf("%s/tpsf%d.txt", datpath, k-1);
    [mHeaders, nMHeaderLines] = parseHeader(bkgPath);
    if (k == 1)
        bkgTpsfVec = zeros(mHeaders.NBINS, nMeasFiles);
        measTpsfVec = zeros(mHeaders.NBINS, nMeasFiles);
    end
    bkgTpsfVec(:,k) = readmatrix(bkgPath, "NumHeaderLines", nMHeaderLines);
    measTpsfVec(:,k) = readmatrix(measPath, "NumHeaderLines", nMHeaderLines);
end
%%
% Load param values
paramName = '';
allfnames = {dir(datpath).name};
for i = 1:length(allfnames)
    if (startsWith(allfnames{i}, "params"))
        if isempty(paramName)
            paramName = allfnames{i};
        else
            warning("Multiple param files");
        end
    end
end

load(sprintf("%s/%s", datpath, paramName));

save(sprintf("%s/%s/%s_training_dat", basedir, datdir, datdir),"bkgTpsfVec", "measTpsfVec",...
    "mHeaders", "nMHeaderLines", "p",...
    "scenes", "rowStrs", "colStrs", "zStrs", "nAbsVec");