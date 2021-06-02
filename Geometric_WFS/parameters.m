addpath ./turbulence
addpath ./tools

diameter = 1; % in metres
%asize = 250;  % in pixels.
radonSize = 499;

fresnelExtent = 2;
angles = 0:22.5:179;
numModes = 20;
%D_r0 = 4;

lambda = 600e-9; % narrowband.
wavelengthList = [600:40:800 820:20:1000]*1e-9; % broadband approximation

%zIndex = [14000 15000 23000 33000 43000 53000 63000 73000 100000 150000 200000];
%z = zDistances(zIndex);

z = 14000;
%z = 20000;
zDistances = [14000 15000 23000 33000 43000 53000 63000 73000 100000 150000 200000];
turbLevels = [0.1 0.2 0.5 1 2 4 8 16 25];

%tel = circle(asize/2, asize);
validRegion = 201:550;
overrideError = 2;


