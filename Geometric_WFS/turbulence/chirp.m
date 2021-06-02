function result = chirp(ax, ay, grid_size);
% RESULT = CHIRP(AX, AY, GRID_SIZE).
% This function generates a 2D chirp with the 
% coefficient ax in the x direction and ay 
% in the y direction.  The final chirp is of size
% GRID_SIZE by GRID_SIZE.  The grid spacing is
% given by deltax.
% Usage: result = chrip(5, 2, 64);
% Created By: Rachel Johnston, 29/5/97

%Initialising Variables
j = sqrt(-1);
result = zeros(grid_size);

%Generating the chirp
for x = 1:grid_size
	for y = 1:grid_size
		result(x,y) = exp(j*ax*((x-grid_size/2)^2))*exp(j*ay*((y-grid_size/2)^2));
	end
end		