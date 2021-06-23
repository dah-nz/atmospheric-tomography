%% 
% Date: 29 Feburary 2020
%
% The following code is (mostly) provided by: Yong Chew from around 2007
% Comments have been provided by Steve Weddell: between 2009 to 2020.
%
% The following are a few comments to try to document the processes 
% that are required to generate firstly two defocused pupil planes
% in a simulated telescope, and secondly, to simulate both a Geometric
% and Curvature wavefront sensor. 
%%

%%
% A few extra "what-if parameters...
%%
function [RMS_act, RMS_curv, RMS_diff] = yongs_demo_with_photon_noise(total_photons)

asize = 250; % was 128
D_r0 = 10;
%Dr0 = D/r0;
tel = circle(asize/2, asize);
% new_asize = 256;
addpath ./../saloni/chap_7;
addpath ./../poisson;

%total_photons;  % normalised at exp(15)

%%
% The header is a bit of of a legacy from Yong's code. It just 
% sets up some parameters, such as the telescope aperture, r0, etc.
%%
old_header

%%
% This is where we create the curvelet wavefront sensor inverse matrix
%%
%curvelet_ForwardMatrix = curvelet_midPoint_ForwardMatrix(phase,lambda, asize, diameter, fresnelExtent, z, validRegion, theseModes);


%%
% Here we create the forward matrix, as used by the geometric wavefront
% sensor, and then the inverse matrix using the pseudoinverse function.
% As shown, the area in which this matrix is serviced is of significance.
% In addition, the modes specified in this argument also define the number
% of Zernikes used for extimating each mode. 
%%
Hmatrix = createGeoWFSMatrix(asize, diameter, fresnelExtent, lambda, z, validRegion, radonSize, angles, numModes);
HmatrixInverse = pinv(Hmatrix' * Hmatrix) * Hmatrix';

%%  Curvature forward and inverse matrix code
% 
%   - Get parmeters from "old_header"
%   - Provide for broadband simulation, i.e, 600-1000 nm
%     for each Zernike mode
%       get z_mode and scale by "pert", i.e, a small number (1e-7)
%       calc a broadband Fresnel propation for that mode for extrafocal
%       calc  "    "        "         "     "    "    "   "  intrafocal
%       calc "signal" by subtracting intra- from extra-focal results
%       narrow the range and put signal in csSignal matrix indexed by mode
%     rescale "pert" over whole matrix
%%
csMatrix = createCurvatureForwardMatrix(wavelengthList, asize, diameter, fresnelExtent, z, validRegion, 2:numModes);

%% Creating an extra validregion for csForward Matrix
%    - not sure why, but I guess the extra space is needed for inverse
%%
csMatrix = [zeros([vlength(validRegion)^2 1]) csMatrix];


%% Create an inverse matrix for the curvature sensor
%   USe pseudoinverse to generate an inverse matrix to "fit"  
%   See Linear Regression notes p. 10-9 in Gorinevsky's notes, 2005.
%%
csMatrixInverse = pinv(csMatrix' * csMatrix) * csMatrix';


%% 
% We need a phase screen to propagate the wavefront through, so...
% generate a phase screen using Rachel Johnson's new midpoint algorithm
%%
%   phase = new_midpoint(asize, D_r0);
%% 
  phase1 = 0.7 * zernike(asize,asize, 2);
  phase2 = -0.3 * zernike(asize,asize, 4);
  phase3 = 0.2 * zernike(asize,asize, 6);
  phase = phase1 + phase2 + phase3;
  
[z1,zx, zy] = createzernikes2(numModes, 250, 1);
[vec_actual,screen_actual]=get_real_vec(phase,z1,numModes);
vec_actual = vec_actual; % .* exp(20) = 3.3970e+05; exp(15) = 2.2328e+03 % was * 1e-1
figure(7), imagesc(phase);


%%
% Next, we generate two defocused pupil planes using Fresnel diffraction
% Generation of the intra and extra focal planes P {fplus, fminus),
% separated by z either side of the focal point. These planes are
% used by both the Geometric and Curvature sensors.

fplus = myFresnelProp(tel.*exp(sqrt(-1)*phase), 1, 2, lambda, z, 0);
fminus = myFresnelProp(tel.*exp(sqrt(-1)*phase), 1, 2, lambda, -z, 0);

ip = abs(fplus(validRegion,validRegion)).^2;
figure(8), imagesc(ip);
im = abs(fminus(validRegion,validRegion)).^2;
figure(9), imagesc(im);
sum(sum(ip))
sum(sum(im))
ip = ip / sum(sum(ip)) .* total_photons;
im = im / sum(sum(im)) .* total_photons;

noisyIm = fast_poisson(im);
figure(28), imagesc(noisyIm);

noisyIp = fast_poisson(ip);
figure(29), imagesc(noisyIp);


signal = (noisyIp - noisyIm); % * 2*pi/lambda;  % this is correct order; don't reverse this!
%figure(10), imagesc(signal);

%%
% Get actual phase after photon noise has been induced for fair comparison
%
%[z1,zx, zy] = createzernikes2(numModes, 350, 1); %as 250
%[vec_actual,screen_actual]=get_real_vec(signal,z1,numModes);  % was phase
%vec_actual = vec_actual; % .* exp(20) = 3.3970e+05; exp(15) = 2.2328e+03 % was * 1e-1
%figure(7), imagesc(signal);


%% Generate or acquire signal for decomposition into Zernikes
%   This is going to generate a single column vector of Zernikes for
%   firstly the Curvature WFS, i.e, "est", and then the Geometric WFS, i.e,
%   "est2".
%   Note that this is effectivelty a matrix multiplication of the
%   pre-derived respective inverse matrices. 
%%
est = csMatrixInverse*signal(:);  % first 20 Zernike coeff for Curvature sensor
figure(1), plot(est); title('Curvature WFS Modes');

est2 =  geoWFS(ip,im, z , HmatrixInverse, angles);
figure(2), plot(est2); title('Geometric WFS Modes');

figure(3), plot(vec_actual); title('WFS Act. coefficients');

%%
 % calculate RMS of each wavefront model result, i.e., actual and curvature
%%
%
geo_mode_diff = vec_actual - est2;
figure(4), plot(geo_mode_diff); title('Geo error plot');

curv_mode_diff = vec_actual - est;
figure(5), plot(curv_mode_diff); title('Curvature error plot');

switch total_photons
   case exp(20)
      est = est .* 5.0735e+07;
   case exp(15)
      est = est .* 1e10-06;
   otherwise
      est = est;
end

Wrms_act = vec_actual; 
partial_Wrms_act = 0;
for j=2:numModes  % modes = 9 i.e., mode1 (tip/tilt), mode2 (defocus/astig_x/astig_y), mode3(coma_x, coma_y_tref_x, tref_y)
   partial_Wrms_act = partial_Wrms_act + Wrms_act(j)^2;
end
WrmstotalAct = sqrt(partial_Wrms_act);    %total rms wavefront error in um
RMS_act = WrmstotalAct;

Wrms_curv = est; 
partial_Wrms_curv = 0;
for j=2:numModes
   partial_Wrms_curv = partial_Wrms_curv + Wrms_curv(j)^2;
end
WrmstotalCurv = sqrt(partial_Wrms_curv);    %total rms wavefront error in um
RMS_curv = WrmstotalCurv;

partial_Wrms_diff = 0;
for j=2:numModes
   partial_Wrms_diff = partial_Wrms_diff + (Wrms_act(j) - Wrms_curv(j))^2;
end
WrmstotalDiff = sqrt(partial_Wrms_diff);    %total rms wavefront error in um
RMS_diff = WrmstotalDiff;


end


