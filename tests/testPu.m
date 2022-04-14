% Test that if we obtain m = P*u, then u == pinv(P)*m

clear; close all; clc;

load('/mnt/data0/yz142/9_19_19_nesterov/paramsP.mat');

VOX_L = 40;
VOX_W = 40;
VOX_H = 1;
ABSVOX_ROW = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18];
ABSVOX_COL = [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26];
ABSVOX_Z = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

u = zeros(VOX_L, VOX_W);
for i = 1:length(ABSVOX_ROW)
    u(ABSVOX_ROW(i), ABSVOX_COL(i)) = 1;
end

uVec = zeros(VOX_L*VOX_W,1);
for row = 0:VOX_L-1
    for col = 0:VOX_W-1
        vecInd = row * VOX_W + col;
        uVec(vecInd + 1) = u(row + 1, col + 1);
    end
end

mFwd = P*uVec;
mNoisy = mFwd  + normrnd(0, 1e-6, size(m));
uVecRecon = pinv(P)*mNoisy;
uRecon = zeros(VOX_L, VOX_W);
for row = 0:VOX_L-1
    for col = 0:VOX_W-1
        vecInd = row * VOX_W + col;
        uRecon(row + 1, col + 1) = uVecRecon(vecInd + 1);
    end
end

imagesc(uRecon);
colormap(flipud(gray));




