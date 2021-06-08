function coef = geoWFS(Ip, Im, z, HmatrixInverse, angles)

% geoWFS Geometric wavefront sensing
%
% function coef = geoWFS(Ip, Im, z, HmatrixInverse, angles);
%
% Performs wavefront sensing using Marcos's geometric wavefront sensor.
% Inputs:
% Ip - Normalised intensity distribution after the aperture.
% Im - Normalised intensity distribution before the aperture.
% z  - Distance of operation.
% HmatrixInverse - Pseudo inverse of the "forward problem". 
%	Hmatrix is generated using createGeoWFSMatrix.m, and
%	HmatrixInverse should be pinv(H' * H)*H'.
% angles - Angles over which the projections are to be taken.
%	Must be the same angles used to construct Hmatrix.
% 
% Outputs:
% coef
%	- Zernike coefficients of the estimated wavefront.
%	Calculated using pinv(Hmatrix'*Hmatrix)*Hmatrix'*data,
%	where data is obtained from the above inputs, ie histogram
%	specification over all projection angles.
%	The number of modes estimated, order of modes, etc are
%	determined by Hmatrix. Refer createGeoWFSMatrix.m
%
%	Use geoWFSSignal instead. This is much preferred because
%	it returns a linear signal to be inverted yourself using
%	HmatrixInverse * geoWFSSignal(Ip,Im, z, angles);
%
% 08-03-2006: Changed the call from radon.m to myradon.m to work
% around Mathwork$' licensing restrictions.
 
numAngles = max(size(angles));

%radonIp = myradon(Ip, angles);
%radonIm = myradon(Im, angles);
% Was: (requires Image processing license)
radonIp = radon(Ip, angles);
radonIm = radon(Im, angles);

radonSize = size(radonIp,1);
if(radonSize*numAngles ~= size(HmatrixInverse,2))
	error(['Invalid HmatrixInverse - it needs to be of size X by ' int2str(radonSize*numAngles) '.']);
end

cumIp = cumsum(radonIp);
cumIm = cumsum(radonIm);

data = zeros([radonSize*numAngles 1]);

for projection = 1:numAngles

	hdiff = horizontalDiff(cumIm(:,projection), cumIp(:,projection));
	data((projection-1)*radonSize+(1:radonSize)) = -hdiff.*radonIm(:,projection) / z;

end

coef = HmatrixInverse * data;

