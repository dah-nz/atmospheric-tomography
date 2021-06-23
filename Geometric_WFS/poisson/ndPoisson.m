function array = ndPoisson(input);

%
% function array = ndPoisson(input);
%
% Wrapper to extend fish.m to N-dimensional arrays
% fish.m is equivalent to poissrnd.m. poisson.m is
% bugged in the original implementation inherited 
% from marcos ('~/mine/marcos');

% Depending on platform/license/release version
% (and the phase of the moon, alignment of planets)
% Matlab might not provide poissrnd (which is faster
% than fish - at least for my version). Whenever poissrnd
% is available, I strongly recommend using it.

%----------
%array = reshape(poissrnd(input(:)), size(input));
%----------
array = reshape(fish(input(:)), size(input));
warning('ndPoisson is calling fish: suggest using poissrnd instead.');
%----------
