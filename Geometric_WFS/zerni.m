function [z] = zerni(x,y,j)

Rnm = 0;
n   = 0;
r = sqrt(x.^2 + y.^2);
teta = atan2(x,y);

while (n*(n+1)/2 - j < 0)
   n = n + 1;
end

n = n - 1;
m = floor((j - n*(n+1)/2 - rem(n,2))/2) * 2 + rem(n,2);

smax  = (n-m)/2;
smaxk = smax + m;

for s = 0:smax
   coef = ((-1).^s)*prod(1:n-s);
   coef = coef ./ (prod(1:s) .* prod(1:(smaxk - s)) .* prod(1:smax - s));
   p = (n- 2*s);
   Rnm = Rnm + coef .* (r.^(p));
end

if m == 0
   z = sqrt(n+1) .* Rnm;
elseif (rem(j,2) == 0)
   z = sqrt(2*(n+1)) .* Rnm .* cos(m* teta);
else
   z = sqrt(2*(n+1)) .* Rnm .* sin(m* teta);
end

