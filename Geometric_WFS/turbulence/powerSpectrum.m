function [real_screen, imag_screen] = powerSpectrum(D_ro, grid_size)
% [REAL_SCREEN, IMAG_SCREEN] = POWERSPECTRUM(D_RO, GRID_SIZE);
% This function generates a Kolmogorov phasescreen
% using the power spectrum method.  The phasescreen
% created is of size GRID_SIZE and is generated for
% the specified D_RO (D/ro).
% Usage: phasescreen = powerSpectrum(20, 128);
% Created By: Cressida Harding and Rachel Johnston

start_time = cputime;

%Initialising variables and arrays
kolmogorov = zeros(grid_size);

%Generating the Kolmogorov power spectrum
%D_ro has been set to zero
for i=1:grid_size
	for j=1:grid_size
		if (i == (grid_size/2 + 1)) & (j == (grid_size/2 + 1))
			%Setting the DC component to zero
			kolmogorov(i,j) = 0;
		else	
			kolmogorov(i,j) = (0.023*((2)^(5/3)))/(((j-grid_size/2-1)^2 + (i-grid_size/2-1)^2)^(11/6));
		end	
	end
end		

%Looking at amplitudes now not powers
kolmogorov_amp = sqrt(kolmogorov);

%Generating a complex random array of size grid_size
random_array = randn(grid_size) + sqrt(-1)*randn(grid_size);

%Generating an random array with the kolmogorov 
%power spectrum
random_kolmogorov = random_array.*kolmogorov_amp;

%Taking the inverse fourier transform so as to
%generate the phase screen
complex_result = (ifft2(fftshift(random_kolmogorov)));

%Taking the real and complex parts - each of which
%is a phase screen and multiplying by (D/ro)^5/6
real_screen = real(complex_result)*((D_ro)^5/6);
imag_screen = imag(complex_result)*((D_ro)^5/6);


%Generating a speckle pattern

%Initialisation
%i = sqrt(-1);

%Generating exp(i*phase_screen)
%eiphi = zeros(grid_size);
%for row = 1:grid_size
%	for col = 1:grid_size
%		eiphi(row,col) = exp(i*real_screen(row,col));
%	end
%end	

%Simulating a square aperture 
%packed = zeros(2*grid_size);
%packed(round(grid_size/2):(round(grid_size/2)+grid_size-1),round(grid_size/2):(round(grid_size/2) + grid_size-1)) = eiphi;	

%Taking the inverse fourier transform
%almost_speckle = fftshift(ifft2(packed));
%Generating the speckle pattern
%speckle = ((abs(almost_speckle)).^2 );

elapsed_time = cputime - start_time