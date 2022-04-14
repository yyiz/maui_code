clear; close all; clc;

rng(1);

addpath(genpath("../../lib/"));

% Initialize simulation parameters
p.CONSTANTS_FILE = "srcdet_1_vox_64x64x32";
p.RUNSTATE = "RUNSMC";
p.NSIMS = 200;
p.SCENE = "RANDSPHERES";
p.ABS_COEFF =  1.0;

configdirPath = "../settings";
readConfig(sprintf("%s/%s.h", configdirPath, p.CONSTANTS_FILE));

[scenes, rowStrs, colStrs, zStrs, nAbsVec] = genScenes(p.SCENE, p.NSIMS, [VOX_W, VOX_L, VOX_H]);

% Set path parameters
logFilePath = "data/log_sim";
currdir = pwd;
mauiDir = "/home/yz142/code-research/maui";

templFname = sprintf("%s/settings/%s.h", mauiDir, p.CONSTANTS_FILE);
constFname = sprintf("%s/inc/constants.h", mauiDir);
constOutPath = sprintf("%s/data/constants_sim", mauiDir);
saveFileStr = p.CONSTANTS_FILE;

%%

save(sprintf("%s/data/params_%s", mauiDir, p.CONSTANTS_FILE), "p",...
    "scenes", "rowStrs", "colStrs", "zStrs", "nAbsVec");

for i = 1:p.NSIMS

SAVENAME = sprintf("""tpsf%d.txt""", i-1);

availGPUs = checkAvailGPU;

while length(availGPUs) < 1
    pause(30);
    availGPUs = checkAvailGPU;
end

gpuid = availGPUs(1) - 1; 

nProcs = 1;
params = {"GPUID", gpuid;...
          "RUNSTATE", p.RUNSTATE;...
          "TPSF_FILENAME", SAVENAME;...
          "ABSVOX_ROW", rowStrs{i};...
          "ABSVOX_COL", colStrs{i};...
          "ABSVOX_Z", zStrs{i};...
          "NUM_ABS", nAbsVec(i);...
          "ABSVOX_UA", p.ABS_COEFF};

cd(mauiDir);
rewriteConst(templFname, constFname, params, 1, nProcs); % pass 1 to paramInd since only 1 set of params is ever passed in
logFname = sprintf("%s_%s_%d.txt", logFilePath, saveFileStr, i);
constSaveFname = sprintf("%s_%s_%d.h", constOutPath, saveFileStr,i);
if exist(logFname, 'file')
    fprintf("Log file exists with same name, overwrite?\n");
    keyboard;
end
system("make clean");
system("make");
system(sprintf("./maui > %s &", logFname));
copyfile(constFname, constSaveFname, 'f');
delete(constFname);
cd(currdir);

fprintf("Launched process %d/%d\n", i, p.NSIMS);
pause(5);
end

function availGPUs = checkAvailGPU
    memThresh = 800e6; % 800 MB
    numGPU = gpuDeviceCount;
    availGPUs = [];
    for i = 1:numGPU
        gpu_i = gpuDevice(i);
        memUsed = gpu_i.TotalMemory - gpu_i.AvailableMemory;
        if memUsed < memThresh
            availGPUs = [availGPUs, i];
        end
    end
end

function [scenes, rowStrs, colStrs, zStrs, nAbsVec] = genScenes(sceneType, nScenes, voxDims)
    scenes = zeros([voxDims, nScenes]);
    rowStrs = cell(nScenes, 1);
    colStrs = cell(nScenes, 1);
    zStrs = cell(nScenes, 1);
    nAbsVec = zeros(nScenes,1);
    
    if (strcmp(sceneType, "RANDSPHERES"))
        for n = 1:nScenes
            scene_i = zeros(voxDims);
            nSphrs = randi([3, 6]);
            [X, Y, Z] = meshgrid(1:voxDims(1), 1:voxDims(2), 1:voxDims(3));
            for k = 1:nSphrs
                currSphrX = voxDims(1)*rand;
                currSphrY = voxDims(2)*rand;
                currSphrZ = voxDims(3)*rand;
                currSphrR = max(1, normrnd(6, 2));
                currSphr = (X-currSphrX).^2 + (Y-currSphrY).^2 + (Z-currSphrZ) .^ 2 < currSphrR.^2;
                scene_i(currSphr) = 1;
            end

            [sceneCols, sceneRows, sceneZ] = ind2sub([voxDims(1), voxDims(2), voxDims(3)], find(scene_i));
            sceneCols = sceneCols - 1;
            sceneRows = sceneRows - 1;
            sceneZ = sceneZ - 1;

            rowStr = "{";
            colStr = "{";
            zStr = "{";

            for i = 1:length(sceneCols)
                if (i == length(sceneCols))
                    rowStr = sprintf("%s%d}", rowStr, sceneRows(i));
                    colStr = sprintf("%s%d}", colStr, sceneCols(i));
                    zStr = sprintf("%s%d}", zStr, sceneZ(i));
                else
                    rowStr = sprintf("%s%d, ", rowStr, sceneRows(i));
                    colStr = sprintf("%s%d, ", colStr, sceneCols(i));
                    zStr = sprintf("%s%d, ", zStr, sceneZ(i));
                end
            end
            scenes(:,:,:,n) = scene_i;
            rowStrs{n} = rowStr;
            colStrs{n} = colStr;
            zStrs{n} = zStr;
            nAbsVec(n) = length(sceneCols);
        end
    else
        fprintf("Invalid scene type");
    end
end

function rewriteConst(templFname, constFname, params, paramInd, nProcs)
    fidR = fopen(templFname);
    fidW = fopen(constFname, 'w');
    nextLine = fgets(fidR);
    while ischar(nextLine)
        copyStr = true;
        if startsWith(nextLine, "#define")
            parseLine = split(nextLine);
            entry = parseLine{2};
            for i = 1:size(params,1)
                if strcmp(entry, params{i,1})
                    paramsEntry = params{i,2};
                    if length(paramsEntry) == nProcs
                        paramVal = string(paramsEntry(paramInd));
                    elseif length(paramsEntry) == 1
                        paramVal = string(paramsEntry);
                    else
                        error("Invalid parameter length");
                    end
                    fprintf(fidW, sprintf("#define %s %s\n", entry, paramVal));
                    copyStr = false;
                end
            end
        end
        if copyStr
            fprintf(fidW, nextLine);
        end
        nextLine = fgets(fidR);
    end
    fclose(fidR);
    fclose(fidW);
end
