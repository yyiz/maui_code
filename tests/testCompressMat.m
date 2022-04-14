Lfac = 4;
Wfac = 3;
Hfac = 2;

AL = 12;
AW = 12;
AH = 12;

compL = size(Ablock,1)/Lfac;
compW = size(Ablock,2)/Wfac;
compZ = size(Ablock,3)/Hfac;

A = 1:12^3;
Ablock = reshape(A, AL, AW, AH);

Acompress = zeros(compL, compW, compZ);
for row = 1:compL
    for col = 1:compW
        for z = 1:compZ
            rowStart = (row - 1)*Lfac + 1;
            rowEnd = (row - 1)*Lfac + Lfac;
            colStart = (col - 1)*Wfac + 1;
            colEnd = (col - 1)*Wfac + Wfac;
            zStart = (z - 1)*Hfac + 1;
            zEnd = (z - 1)*Hfac + Hfac;
            sumBlock = Ablock(rowStart:rowEnd, colStart:colEnd, zStart:zEnd);
            Acompress(row, col, z) = sum(sumBlock(:));
        end
    end
end

Avec = zeros(1, AL*AW*AH);
for row = 0:AL-1
    for col = 0:AW - 1
        for z = 0:AH - 1
            vecInd = (z * AL * AW) + (row * AW) + col + 1;
            Avec(vecInd) = Ablock(row + 1, col + 1, z + 1);
        end
    end
end

compressed = compressMat(Avec, AL, AW, AH, Lfac, Wfac, Hfac);
compressedBlock = zeros(size(Acompress));

for row = 0:compL-1
    for col = 0:compW-1
        for z = 0:compZ-1
            vecInd = (z * compL * compW) + (row * compW) + col + 1;
            compressedBlock(row+1, col+1, z+1) = compressed(vecInd);
        end
    end
end

diff = compressedBlock - Acompress;
err = abs(sum(diff(:)));
fprintf("Error: %d\n", err);

