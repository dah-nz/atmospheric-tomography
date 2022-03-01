addpath ./turbulence
addpath ./tools

% Basic parameters
asize = 256;            % pixels. Aperture size
r_0 = 0.05;             % meters. Fried paramter - strength of turbulence
tel = circle(asize/2, asize);
new_asize = 256;

% Extra parameters
diameter = 1;           % in metres
radonSize = 497;        % ?
D_r0 = diameter / r_0;  % ?

fresnelExtent = 2;      % ?
angles = 0:22.5:179;    % in degrees
numModes = 20;          % number of Zernike modes

% Wavelength information
lambda = 600e-9;        % metres. Narrowband wavelength
wavelengthList = [600:40:800 820:20:1000]*1e-9; % metres. Broadband approximation

% Altitudes for turbulence
z = 14000;
zDistances = [14000 15000 23000 33000 43000 53000 63000 73000 100000 150000 200000];
turbLevels = [0.1 0.2 0.5 1 2 4 8 16 25];

validRegion = 201:550;
overrideError = 2;


