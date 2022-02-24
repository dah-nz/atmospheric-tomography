%% 
% Date: 8 June 2021
%
% The following code is (mostly) provided by: Yong Chew from around 2007
% Comments have been provided by Steve Weddell: between 2009 to 2020.
%
% The following are a few comments to try to document the processes 
% that are required to generate firstly two defocused pupil planes
% in a simulated telescope, and secondly, to simulate both a Geometric
% and Curvature wavefront sensor. 
%%

close all

%%
% Set up parameters, such as the telescope aperture, r0, etc.
%%
parameters

%%
% Here we create the forward matrix, as used by the geometric wavefront 
% sensor, and then the inverse matrix using the pseudoinverse function.
% As shown, the area in which this matrix is serviced is of significance.
% In addition, the modes specified in this argument also define the number
% of Zernikes used for extimating each mode. 
%%
Hmatrix = createGeoWFSMatrix(asize, diameter, fresnelExtent, lambda, z, validRegion, radonSize, angles, numModes);
HmatrixInverse = pinv(Hmatrix' * Hmatrix) * Hmatrix';

%% 
% We need a phase screen to propagate the wavefront through, so...
% generate a phase screen using Rachel Johnson's new midpoint algorithm
%%
for mode  = 1:numModes
    % phase = phase_screen(asize, D_r0);
    phase = zernike(asize, asize, mode);


    %%
    % Next, we generate two defocused pupil planes using Fresnel diffraction
    % Generation of the intra and extra focal planes P {fplus, fminus),
    % separated by distance z, measured either side of the focal point. These planes are
    % used are used by the Geometric WFS sensors and in effect are defocued sensors that
    % can be represented in the lab or on-sky by a pair of CCD or sCMOS detector planes, 
    % simply by using two cameras at a distance of 'z' either side of a focal point from 
    % a simple plano-convex lens, of focal point 'f'.
    %%
    field = tel.*exp(sqrt(-1)*phase);
    
    fplus = fresnelPropagation(field, 1, 2, lambda, z, 0);
    figure(mode * 10 + 1), imagesc(abs(fplus).^2); title('Image of the Extra-focal plane');
    fminus = fresnelPropagation(field, 1, 2, lambda, -z, 0);
    figure(mode * 10 + 2), imagesc(abs(fminus).^2);  title('Image of the Intra-focal plane')

    ip = abs(fplus(validRegion,validRegion)).^2;
    im = abs(fminus(validRegion,validRegion)).^2;
    ip = ip / sum(sum(ip));
    im = im / sum(sum(im));

    %signal = ip - im;  % this is correct order; don't reverse this!


    %% Generate or acquire signal for decomposition into Zernikes
    %   This is going to generate a single column vector of Zernikes for
    %    the Geometric WFS, i.e, "est2".
    %   Note that this is effectivelty a matrix multiplication of the
    %   pre-derived respective inverse matrices with an unknown projected wavefront. 
    %%
    est2 =  geoWFS(ip,im, z , HmatrixInverse, angles);
    est2 = est2 / max(abs(est2));
    figure(mode * 10 + 3), plot(est2); title('Geometric WFS Modes');

end




