function field = myFresnelProp(fieldStart, fieldSize, fresnelExtent, lambda, z, overrideError)

% myFresnelProp: Fresnel propagation.
%
% function field = myFresnelProp(fieldStart, fieldSize, fresnelExtent, lambda, z, overrideError)
%
% Fresnel propagation for a finite aperture to a certain finite distance.
%
% Inputs:
% fieldStart: Complex field at the aperture (magnitude * exp(i * phi))
% fieldSize: Size of the aperture in distance units (nominally metres)
% fresnelExtent: Extent of the fresnel kernel to convolve with, in terms
%	of the number of times the size of the aperture (fieldStart).
% lambda: Wavelength of light to propagate with.
% z: distance to propagate to.
% overrideError: Override sampling criterion (you can usually push this
%	for a little bit less than the minimum propagation distance, as
%	given by sampling criteria (phase difference between neighbouring
%	pixels at most pi) .
%
% Output:
% field: Complex field at the desired propagation distance.
%


k = 2*pi / lambda;
numPixels = size(fieldStart,1);


% Phase difference from 1 pixel to the next
% has to be less than pi, or thinking another way,
% 1/(2pi) d/dx( phase )  = instantaneous frequency.
% Fsampling > 2*maxFrequency.
if (abs(lambda * z) <= (fresnelExtent*fieldSize^2/numPixels))
	if (overrideError ~= 1)
		error(['Discretisation error: need to propagate further, or make pixels smaller: lambda * z = ' num2str(lambda*z) ', needs to be greater than ' num2str(fresnelExtent*fieldSize^2/numPixels) '.']);
	end;
end;


temp = ones([round(fresnelExtent*numPixels)+1 1])* (   ((-fresnelExtent*numPixels-1)/2):((fresnelExtent*numPixels-1)/2)   );
temp = temp / numPixels * fieldSize;

fresnelPhase = k / 2 / z * (temp.^2 + (temp').^2);
fresnelKernel = exp(sqrt(-1) * fresnelPhase);


field = cconv2(fresnelKernel, fieldStart);
