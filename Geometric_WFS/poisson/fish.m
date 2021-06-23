% Function to generate Poisson noise.
%
% If the mean mu is very large, then a normal approximation is used
%
function [noise] = poisson(mu)

value = exp(-mu);
noise = zeros(size(mu));

current = rand(size(mu));

while(sum(sum(current > value)) ~= 0)
     noise = noise + (current > value);
     current = current .* rand(size(mu));
end

for d1=1:size(mu,1)
	for d2=1:size(mu,2)
		if mu(d1,d2)>100
			noise(d1,d2)=mu(d1,d2)+round(sqrt(mu(d1,d2))*randn(1));
		end	 
	end
end
		