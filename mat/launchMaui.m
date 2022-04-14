clear; close all; clc;

configFname = "one_shot_dense_multilayer";
saveFileStr = configFname;

% nProcs = 1;
% params = {"GPUID", [0];...
%           "RUNSTATE", ["RUNSMC"];...
%           "TPSF_FILENAME", ["""tpsf0.txt"""];...
%           "HMAP_FILENAME", ["""hmap0.csv"""]};

% nProcs = 4;
% params = {"GPUID", [4, 5, 6, 7];...
% %           "RUNSTATE", ["RUNJAC", "RUNJAC", "RUNJAC", "RUNJAC"];...
%           "RUNSTATE", ["RUNSMC", "RUNSMC", "RUNSMC", "RUNSMC"];...
%           "TPSF_FILENAME", ["""tpsf4.txt""", """tpsf5.txt""", """tpsf6.txt""", """tpsf7.txt"""];...
%           "HMAP_FILENAME", ["""hmap0.csv""", """hmap1.csv""", """hmap2.csv""", """hmap3.csv"""]};

nProcs = 8;
params = {"GPUID", [0, 1, 2, 3, 4, 5, 6, 7];...
          "RUNSTATE", ["RUNJAC", "RUNJAC", "RUNJAC", "RUNJAC",...
                       "RUNSMC", "RUNSMC", "RUNSMC", "RUNSMC",];...
          "TPSF_FILENAME", ["""tpsf4.txt""", """tpsf5.txt""", """tpsf6.txt""", """tpsf7.txt""",...
                            """tpsf0.txt""", """tpsf1.txt""", """tpsf2.txt""", """tpsf3.txt"""];...
          "HMAP_FILENAME", ["""hmap0.csv""", """hmap1.csv""", """hmap2.csv""", """hmap3.csv""",...
                            """hmap4.csv""", """hmap5.csv""", """hmap6.csv""", """hmap7.csv"""]};  
            
                   
logFilePath = "data/log_sim";
currdir = pwd;
mauiDir = "/home/yz142/code-research/maui";
cd(mauiDir);

templFname = sprintf("%s/settings/%s.h", mauiDir, configFname);
constFname = sprintf("%s/inc/constants.h", mauiDir);
constOutPath = sprintf("%s/data/constants_sim", mauiDir);

for i = 0:(nProcs-1)
    rewriteConst(templFname, constFname, params, i+1, nProcs);
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
    fprintf("Launched simulation %d/%d\n", i+1, nProcs);
end
delete(constFname);
cd(currdir);

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