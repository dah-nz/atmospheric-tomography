% Function to generate Poisson noise
% Has to be modified because for high photon counts there is a 
% numerical underflow problem and the Gaussian approximation is required.
%
% Richard Lane November 29th 2000
%
function [noise] = poisson(mean)

value = exp(-mean);
mask = mean < 50;
%
% The mask is applied at a photon count of 50 above this the Gaussian 
% approximation is employed
%
noise = zeros(size(mean));
%
current = rand(size(mean));

while(sum(sum((current > value).*mask)) ~= 0)
     noise = noise + (current > value);
     current = current .* rand(size(mean));
  end
  
noise = round (mean + randn(size(mean)).*sqrt(mean)).*(~mask)+ noise;