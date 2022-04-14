clear; close all; clc;

rng(0);

numSrcSens = 10;
NBINS = 10;
NVOX = 4;
K = 3;

P = rand(numSrcSens*NBINS, NVOX);
m = rand(numSrcSens*NBINS,1);

tempM = m;

MAT_M = size(P, 1);

reducK = floor(NBINS/K);

reducMat = sparse(reducK, MAT_M);
for i = 0:numSrcSens-1
    for j = 0:reducK-1
        reducMat((i*reducK)+j+1, (i*NBINS)+(j*K)+1:(i*NBINS)+(j+1)*K) = ones(1, K);
    end
end

checkReducMat = full(reducMat);
P = reducMat * P;
m = reducMat * m;

% reducP = zeros(reducK, size(P,2));
% reducM = zeros(reducK, 1);
% 
% for i = 0:numSrcSens-1
%     for j = 0:reducK-1
%         reducP((i*reducK)+j+1, :) = sum(P((i*NBINS)+(j*K)+1:(i*NBINS)+(j+1)*K, :),1);
%         reducM((i*reducK)+j+1) = sum(m((i*NBINS)+(j*K)+1:(i*NBINS)+(j+1)*K));
%     end
% end
% P = reducP;
% m = reducM;

