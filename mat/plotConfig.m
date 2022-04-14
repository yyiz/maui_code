clear; close all; clc;

configFname = "one_shot_dense_multilayer";
plotHd = true;


showSrcSens = true;
showAbs = true;
setView = [0, 90];

%%

addpath(genpath("../../lib/"));

configdirPath = "../settings";
readConfig(sprintf("%s/%s.h", configdirPath, configFname));

truthFig = figure();
h1 = axes;
set(h1, 'Zdir', 'reverse');
axis equal;
view(0,0);
xlabel('x (mm)');
ylabel('y (mm)');
zlabel('z (mm)');

% Display heatmap
allZeros = zeros(VOX_L, VOX_W, VOX_H);
xBounds = [VOX_ORIGX, VOX_ORIGX + VOX_W*VOX_SIDELEN];
yBounds = [VOX_ORIGY, VOX_ORIGY + VOX_L*VOX_SIDELEN];
zBounds = [VOX_ORIGZ, VOX_ORIGZ + VOX_H*VOX_ZLEN];

absMap = zeros(VOX_L, VOX_W, VOX_H);
if (showAbs)
    for i = 1:NUM_ABS
        row = ABSVOX_ROW(i); 
        col = ABSVOX_COL(i);
        z = ABSVOX_Z(i);
        absMap(row + 1, col + 1, z + 1) = 1;
        if (plotHd)
            voxel([VOX_ORIGX+col*VOX_SIDELEN, VOX_ORIGY+row*VOX_SIDELEN, VOX_ORIGZ+z*VOX_ZLEN],[VOX_SIDELEN, VOX_SIDELEN, VOX_ZLEN],'k');
        end
    end
end

if (plotHd)
    vol3d('CData', allZeros, 'Alpha', allZeros, 'XData', xBounds, 'YData', yBounds, 'ZData', zBounds, 'EdgeAlpha', 0.1);
else
    allOnes = ones(VOX_L, VOX_W, VOX_H);
    vol3d('CData', allOnes, 'Alpha', absMap, 'XData', xBounds, 'YData', yBounds, 'ZData', zBounds, 'EdgeAlpha', 0.1);
end

colormap(flipud(gray));
caxis([0 1])

% title('Ground Truth');
numTicks = 11;
colorbar('Ticks', linspace(0,1,numTicks), 'TickLabels', linspace(0,ABSVOX_UA,numTicks));
fclose('all');
view(setView);

for i = 1:NUMLAYERS
    if (mod(i, 2) == 1)
        layerAlph = 0.1;
    else
        layerAlph = 0.05;
    end
    layerZ1 = ALL_LAYERS(i);
    layerZ2 = ALL_LAYERS(i + 1);
    layerDx = SLAB_MAXX - SLAB_MINX;
    layerDy = SLAB_MAXY - SLAB_MINY;
    layerDz = layerZ2 - layerZ1;
    voxel([SLAB_MINX, SLAB_MINY, layerZ1],[layerDx, layerDy, layerDz],'k',layerAlph);
end

if (showSrcSens)
    % Plot sources
    srcZ = SRC_ORIGZ;
    for srcI = 0:SRC_L-1
        for srcJ = 0:SRC_W-1
            srcX = srcJ*SRC_SEP + SRC_ORIGX;
            srcY = srcI*SRC_SEP + SRC_ORIGY;
            hold on;
            scatter3(srcX, srcY, srcZ, 20, 'r', 'filled')
        end
    end

for sensI = 0:SENS_L-1
    for sensJ = 0:SENS_W-1
        plotSensX = SENS_ORIGX + sensJ*SENS_SEP;
        plotSensY = SENS_ORIGY + sensI*SENS_SEP;
        plotSensXArr = [plotSensX; plotSensX; plotSensX + SENS_PIXW; plotSensX + SENS_PIXW];
        plotSensYArr = [plotSensY; plotSensY + SENS_PIXW; plotSensY + SENS_PIXW; plotSensY];
        plotSensZArr = [SENS_ORIGZ; SENS_ORIGZ; SENS_ORIGZ; SENS_ORIGZ];
%         rectangle('Position', [plotSensX, plotSensY, SENS_PIXW, SENS_PIXW], 'FaceColor', 'blue');
        patch(plotSensXArr, plotSensYArr, plotSensZArr, 'blue');
    end
end
end
