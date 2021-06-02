function [result] = cconv2(h_est, f_est, extent);

% RESULT = RCONV2(H_EST, F_EST);
% This function performs the convolution
% of H_EST and F_EST in the frequency
% domain - hence it is a multiplication.
% Usage: c = rconv2(a, b);
% Created By: Rachel Johnston, 10/11/97
% Modified by Marcos to choose extent of the answer
% full or valid

dimh = size(h_est, 1);
dimf = size(f_est, 1);
dimg = dimh + dimf - 1;
dimvalid=dimh-dimf+1;

if dimvalid<1
   fprintf('Place the largest matrix first')
end

dimr = 2^nextpow2(dimg);

zeropack_h = zeros(dimr);
zeropack_f = zeros(dimr);
zeropack_h(1:dimh, 1:dimh) = h_est;
zeropack_f(1:dimf, 1:dimf) = f_est;

pack_h_ft = fft2(zeropack_h);
pack_f_ft = fft2(zeropack_f);

result_ft = (ifft2(pack_h_ft.*pack_f_ft));

if (nargin < 3)
    extent = 'f';
end

if extent=='f'
	result = result_ft(1:dimg,1:dimg);
end

if extent=='v'
      coords=0:dimvalid-1;
      result=result_ft(dimf+coords,dimf+coords);
end   

%if inputs are real, so is the convolution, so remove small imaginary parts

if real(h_est)==h_est
	if real(f_est)==f_est
		result=real(result);
	end
end		

