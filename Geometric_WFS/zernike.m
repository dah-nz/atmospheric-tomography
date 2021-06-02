function [zernicoef] = zernike(dim,diameter,order)

% A function to calculate Zernike coefficients in a unit circle.
%
%
% 	[zernicoef] = zernike(dim,diameter,order)
%
% Program included: zernim.m

% by : R. Irwan & R.G. Lane
% *****************************************************************************
% Modified by Marcos van Dam to replace the loops with
% matrix calculations

N = (dim+1)/2;
radius = diameter/2;
sq_radius = radius.^2;

k = (-N:N-1)'*ones(1,dim+1);
l = ones(dim+1,1)*(-N:N-1);
  
sq_distance = k.^2 + l.^2;

x = l/radius;
y = k/radius;
zernicoef = zerni(x,y,order);

zernicoef=zernicoef.*(sq_distance<=sq_radius); %only valued within the unit circle
zernicoef = zernicoef(2:dim+1,2:dim+1);
