%%
% Inputs
% asize         - (pixels) aperture size in pixels
% diameter      - (m) telescope diameter 
% fresnelExtent - ?
% lambda        - (m) initial wavelength
% z             - (m) altitude
% validRange    - ?
% radonSize     - ?
% angles        - degrees
% numModes      - number of Zernike modes to simulate

% Outputs
% Hmatrix       - matrix of Geometric WFS
%%
function Hmatrix = createGeoWFSMatrix(asize, diameter, fresnelExtent, lambda, z, validRange, radonSize, angles, numModes)

numAngles = max(size(angles));

Hmatrix = zeros([radonSize*numAngles numModes]);

overrideError = 1;

%---------------------------------
for mode = 1:numModes


    disp(['Calculating mode ' int2str(mode) ' of ' int2str(numModes) '.']);

    randomWavefront = zernike(asize,asize,mode)*1e-8; %make sure it is really small.

    Ip = fresnelPropagation(circle(asize/2,asize).*exp(sqrt(-1)* randomWavefront*2*pi/lambda), diameter, fresnelExtent, lambda, z, overrideError);
    Im = fresnelPropagation(circle(asize/2,asize).*exp(sqrt(-1)* randomWavefront*2*pi/lambda), diameter, fresnelExtent, lambda, -z, overrideError);
    Ip = abs(Ip).^2;
    Im = abs(Im).^2;

    % I = circle(asize/2,asize);
    % I = I / sum(sum(I));
    
    Ip = Ip(validRange, validRange);
    Ip = Ip / sum(sum(Ip));
    
    Im = Im(validRange, validRange);
    Im = Im / sum(sum(Im));

    [radonIp, Ixp] = radon(Ip,angles);
    [radonIm, Mxp] = radon(Im,angles);
    %%
    % Here, the CDF is calcuated for both the intra and extra pupil planes.
    % Refer to van Damm's PhD thesis, partcularly Fig. 10.7.
    %%
    cumIp = cumsum(radonIp);
    cumIm = cumsum(radonIm);

    %%
    % Note the scaling performed before assigning to "randomWavefront". There is still much discussion in our group as to why this is done
    % For example, it doesn't affect the radon or CDF results in the slightest.
    % However, it may affect the histogram calcuation called here.
    %%
    for projection = 1:numAngles
        hdiff = horizontalDiff(cumIm(:,projection), cumIp(:,projection));
        Hmatrix((projection-1)*radonSize+(1:radonSize), mode) = -hdiff.*radonIm(:,projection) / z;	
    end


end  % of mode histogram generation.
%-----------------------------
Hmatrix = Hmatrix / 1e-8;
