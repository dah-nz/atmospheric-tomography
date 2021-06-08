function [screen] = phase_screen(grid_size, D_ro)
%  Simulation of a phase screen with Kolmogorv Statistics.
%  Starts from an exact 15 x 15 phase screen.
%
%  function [screen] = new_midpoint(grid_size, D_ro);
% 
%  Inputs	min_grid_size	- minimum desired phase screen size
%           D_ro            - D/r0 value, determines severity of turbulence
%
%  Outputs	screen          - a phase screen with Kolmogorov Statistics
%                             of size [min_grid_size, min_grid_size]
%
%  Requires phase44.m, svectors4.mat
%
%
%  Rachel Johnston and Cressida Harding
%


%Finding the number of increases required
%Starting from a phasescreen of 15 by 15
calculated_size  = 15;
n_increases = 0;
while (calculated_size < grid_size)
	calculated_size = 2*calculated_size - 11;
	n_increases = n_increases + 1;
end

%Generating the phasescreen
full_screen = feval('phase44', n_increases);
chopped_screen = full_screen(1:grid_size, 1:grid_size);

%Need to rescale the screen so that it has the appropriate D_ro after
%the chopping
size_chopped = size(chopped_screen,1);
size_full = size(full_screen,1);
scale = ((size_full - 1)/(size_chopped - 1))^(5/6);
scaled_screen = chopped_screen*scale;

%Scaling so it has the specfied D_ro
screen = scaled_screen*((D_ro)^(5/6));

