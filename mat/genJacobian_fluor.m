clear; close all; clc;

Jdir = "3_9_22_dipendot_jacobian";

tau = 1000;


if (gpuDeviceCount > 1)
    defaultPath = "/mnt/data0/yz142";
else
    defaultPath = "D:/results_data";
end

loadname_J = sprintf("%s/%s/J.mat", defaultPath, Jdir);
savename_J = sprintf("%s/%s/J_fluor.mat", defaultPath, Jdir);
load(loadname_J);
addpath(genpath("../../lib/"));

NBINS_orig = Jheaders.NBINS;
binW = (Jheaders.TIME_MAX - Jheaders.TIME_MIN) / Jheaders.NBINS;
J = reshape(J, Jheaders.NBINS, []);
[J, Jheaders.timeAx] = convTpsf(J, 'exp', binW, 'lifetime', tau, 'multicol', true);

Jheaders.NBINS = length(Jheaders.timeAx);
NBINS_fluor = Jheaders.NBINS;
J = reshape(J, [], prod([Jheaders.VOX_L, Jheaders.VOX_W, Jheaders.VOX_H]));

save(savename_J, "-v7.3", "J", "Jheaders", "nJHeaderLines", "tau");
fprintf("Done saving Jacobian variables to: %s\n", savename_J);

loadname_bkg = sprintf("%s/%s/tpsf.mat", defaultPath, Jdir);
if exist(loadname_bkg, 'file')
    load(loadname_bkg);
    mHeaders.NBINS = NBINS_fluor;
    nZeros = NBINS_fluor - NBINS_orig;
    bkgTpsf = [bkgTpsf; zeros(nZeros,1)];
    diffTpsf = [diffTpsf; zeros(nZeros,1)];
    perturbTpsf = [perturbTpsf; zeros(nZeros,1)];
    savename_bkg = sprintf("%s/%s/tpsf_fluor.mat", defaultPath, Jdir);
    save(savename_bkg, "-v7.3", "mHeaders", "nMHeaderLines", "bkgTpsf", "perturbTpsf", "diffTpsf");
    fprintf("Done saving bkg tpsf variables to: %s\n", savename_bkg);

else
    fprintf("No bkg tpsf file\n");
end




