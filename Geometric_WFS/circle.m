%
% File to generate a circular array of ones in a square array
%
%  function [image] = circle(radius, dimension)
%

function [image] = circle(radius, dimension)
%
image=zeros(dimension,dimension);

sqr_radius = radius^2;
centre = dimension/2+0.5;
for i=1:dimension
    for j=1:dimension
        if ((i-centre)^2 + (j-centre)^2 <= sqr_radius) 
            image(i,j) = 1.0;
            end
        end
    end


    
