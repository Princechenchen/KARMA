function y = Michalewicz_3d(xx)
%[0,pi]
if (nargin == 1)
    m = 10;
end

d = size(xx,2);
sum = 0;

for ii = 1:d
	xi = xx(:,ii);
	new = sin(xi) .* (sin(ii.*xi.^2./pi)).^(2.*m);
	sum  = sum + new;
end

y = -sum;
end