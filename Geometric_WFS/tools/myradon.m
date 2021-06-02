function P=myradon(im, theta)

% Optimised version of myradonc.m, which is a direct translation
% from C, and is kept archived for reference.
% As of 08-03-06, no optimisation has been carried out (yet?),
% we might not need to... 
% 

im = double(im);

numAngles = prod(size(theta));
radTheta = theta * pi / 180; % this originally called thetaPtr

% see the for loop below: radonc iterates with M inside N.
% suggesting that M goes down rows, and N across columns.
% See also the notation in "help size"
[M,N] = size(im);
xOrigin = max(0, (floor((N-1)/2))); % be careful with integer division in C -> (N-1)/2
yOrigin = max(0, (floor((M-1)/2)));

temp1 = double(M - 1 - yOrigin);
temp2 = double(N - 1 - xOrigin);

% From "help radon"
% Size of output should be:
%        2*ceil(norm(size(I)-floor((size(I)-1)/2)-1))+3
rLast = (ceil(sqrt(temp1^2 + temp2^2))) + 1;
rFirst= - rLast;
rSize = rLast - rFirst + 1;

P = zeros([rSize numAngles]);

% For debugging.
% disp([M N xOrigin yOrigin temp1 temp2 rLast rFirst rSize size(P)]);pause;

% --------- static void radon ... -----------

xCosTable = zeros([2*N 1]);
yCosTable = zeros([2*M 1]);

for k = 1:numAngles
    angle = radTheta(k);
     
    cosine = cos(angle);
    sine   = sin(angle);

    for n = 1:N
        x = (n - 1) - xOrigin; % -1 for array index offset
        xCosTable(2*n-1) = (double(x) - 0.25) * cosine;
	xCosTable(2*n) = (double(x) + 0.25) * cosine;
    end;
    
    for m = 1:M
	y = yOrigin - (m - 1); % -1 for array index offset
	ySinTable(2*m - 1) = (double(y) - 0.25)*sine;
	ySinTable(2*m) = (double(y) + 0.25)*sine;
    end;


    pixelIndex = 1;

    for n = 1:N
	for m = 1:M
	    pixel = im(m,n);

	    if (pixel ~= 0.0)
		pixel = pixel * 0.25;

		r = xCosTable(2*n-1) + ySinTable(2*m-1) - rFirst;

		% incrementRadon is inlined each time.
		%incrementRadon
		r1 = floor(r);
		delta = r - r1;

                % + 1 to array base index in Matlab.
		P(r1+1,k) = P(r1+1,k) + pixel * (1 - delta);
		P(r1+2,k) = P(r1+2,k) + pixel * delta;

		r = xCosTable(2*n) + ySinTable(2*m-1) - rFirst;
		%incrementRadon
		r1 = floor(r);
		delta = r - r1;
                % + 1 to array base index in Matlab.
		P(r1+1,k) = P(r1+1,k) + pixel * (1 - delta);
		P(r1+2,k) = P(r1+2,k) + pixel * delta;



		r = xCosTable(2*n-1) + ySinTable(2*m) - rFirst;
		%incrementRadon
		r1 = floor(r);
		delta = r - r1;
                % + 1 to array base index in Matlab.
		P(r1+1,k) = P(r1+1,k) + pixel * (1 - delta);
		P(r1+2,k) = P(r1+2,k) + pixel * delta;


		r = xCosTable(2*n) + ySinTable(2*m) - rFirst;
		%incrementRadon
		r1 = floor(r);
		delta = r - r1;
                % + 1 to array base index in Matlab.
		P(r1+1,k) = P(r1+1,k) + pixel * (1 - delta);
		P(r1+2,k) = P(r1+2,k) + pixel * delta;

	    end;
	end;
    end;
end;



	

