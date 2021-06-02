function hdiff = horizontalDiff(cumI1, cumI2)

% horizontalDiff - Histogram specification for GWS
%
%function hdiff = horizontalDiff(cumI1, cumI2);
%
% Given 2 cumulative intensity functions, horizontalDiff
% will return the horizontal differences as is needed for
% geometric wavefront sensing. The array returned will be
% a column vector of the same size as the inputs, and will
% contain the differences between the same coordinates.
% ("Difference" from cumI1 to cumI2)

hdiff = zeros([size(cumI1,1) 1]);
for index = 1:size(cumI1,1)
	bitMore = min(find(cumI2(:) >= cumI1(index)));  
    % find all the nonzero values in the 2D array "cumI2" that are greater than or 
    % equal to tne value in "cumI1" at array "index", and return the minimum of these values.
    
	if (size(bitMore) == [0 1])  % if the min is in 0 or 1, then
		bitMore = size(cumI1,1); %   that size element becomes "bitMore"
	end;
    
	bitLess = max(find(cumI2(:) <= cumI1(index)));
    % find all the nonzero values in the 2D array "cumI2" that are less than or 
    % equal to tne value in "cumI1" at array "index", and return the maximum of these values.
    
	if (size(bitLess) == [0 1])
		bitLess = 1;
	end;
	
	if (bitMore <= bitLess)
		hdiff(index) = 0;
	else
		hdiff(index) = bitLess - index + (cumI1(index)-cumI2(bitLess)) / (cumI2(bitMore) - cumI2(bitLess)) * (bitMore - bitLess);
	end;
end;
end

