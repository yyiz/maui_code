clear; close all; clc;

useTextScan = true;

saveJ = true;
saveM = true;

if (gpuDeviceCount > 1)
    defaultPath = "/mnt/data0/yz142";
else
    defaultPath = "D:/results_data";
end

basedir = "3_13_22_one_shot_tofdot";

jFpath = sprintf("%s/%s/tempfiles", defaultPath, basedir);
jFnum = [0, 1, 2, 3];
jDestname = sprintf("%s/%s/J.mat", defaultPath, basedir);

mFpath = sprintf("%s/%s/tempfiles", defaultPath, basedir);
mFnum = [0, 1, 2, 3];
mDestname = sprintf("%s/%s/tpsf.mat", defaultPath, basedir);

%%
addpath(genpath("../../lib/"));

% Read in jacobian J
if (saveJ)
    fprintf("Loading Jacobian files...\n");
    jIterNum = 0;
    J = 0;
    for i = 1:length(jFnum)
        Jpath = sprintf("%s/hmap%d.csv", jFpath, jFnum(i));
        [Jheaders, nJHeaderLines] = parseHeader(Jpath);
        if (useTextScan)
            if (i == 1)
                nrows = prod([Jheaders.NBINS, Jheaders.SRC_L, Jheaders.SRC_W, Jheaders.SENS_L, Jheaders.SENS_W]);
                ncols = prod([Jheaders.VOX_L, Jheaders.VOX_W, Jheaders.VOX_H]);
                fmtRow = repmat('%f, ', 1, ncols-1); fmtRow = strcat(fmtRow, '%f\n');
            end
            Jfid = fopen(Jpath);
            J = J + cell2mat(textscan(Jfid, fmtRow, 'Headerlines', 31));
            fclose(Jfid);
        else
            J = J + readmatrix(Jpath, "NumHeaderLines", nJHeaderLines);
        end
        jIterNum = jIterNum + Jheaders.NUM_ITERS;
        fprintf("Done with %d/%d\n", i, length(jFnum));
    end
    Jheaders.NUM_ITERS = jIterNum;
    if (isfield(Jheaders, "NORMALIZETPSF") && Jheaders.NORMALIZETPSF == 0)
        normalize = Jheaders.NUM_SAMPS * Jheaders.NUM_ITERS;
    else
        normalize = length(jFnum);
    end
    J = J ./ normalize;
    if (exist(jDestname, 'file') == 2)
        fprintf("Jacobian destination file already exists. Continue to overwrite\n");
        keyboard;
    end

    save(jDestname, "-v7.3", "J", "Jheaders", "nJHeaderLines");
    fprintf("Done saving Jacobian variables to: %s\n", jDestname);
end

if (saveM)
    fprintf("Loading measurement files...\n");
    % Read in measurement m
    mIterNum = 0;
    bkgTpsf = 0;
    perturbTpsf = 0;
    for i = 1:length(mFnum)
        Mpath = sprintf("%s/tpsf%d.txt", mFpath, mFnum(i));
        BkgPath = sprintf("%s/background_tpsf%d.txt", mFpath, mFnum(i));
        [mHeaders, nMHeaderLines] = parseHeader(Mpath);
        mIterNum = mIterNum + mHeaders.NUM_ITERS;
        bkgTpsf = bkgTpsf + readmatrix(BkgPath, "NumHeaderLines", nMHeaderLines);
        perturbTpsf = perturbTpsf + readmatrix(Mpath, "NumHeaderLines", nMHeaderLines);
    end
    mHeaders.NUM_ITERS = mIterNum;
    if (isfield(mHeaders, "NORMALIZETPSF") && mHeaders.NORMALIZETPSF == 0)
        normalize = mHeaders.NUM_SAMPS * mHeaders.NUM_ITERS;
    else
        normalize = length(mFnum);
    end
    bkgTpsf = bkgTpsf ./ normalize;
    perturbTpsf = perturbTpsf ./ normalize;
    diffTpsf = bkgTpsf - perturbTpsf;
    if (exist(mDestname, 'file') == 2)
        fprintf("Measurement destination file already exists. Continue to overwrite\n");
        keyboard;
    end
    save(mDestname, "-v7.3", "mHeaders", "nMHeaderLines", "bkgTpsf", "perturbTpsf", "diffTpsf");
    fprintf("Done saving tpsf to %s\n", mDestname);
end

