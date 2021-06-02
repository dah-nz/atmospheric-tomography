function [ro] = cn2h2ro(cn2h, lambda)
%This function takes the atmospheric strength and wavelength of light
%and returns ro , Fried's parameter, in m.

k = 2*pi/lambda;
ro = (0.423*k^2*cn2h)^(-3/5); 	%gives ro in metres

