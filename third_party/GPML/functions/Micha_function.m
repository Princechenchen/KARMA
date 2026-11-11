function y = Micha_function(xx)
xx=xx./4.*2.5;
if (nargin == 1)
    m = 4;
end

d = size(xx,2);
sum = 0;
for ii = 1:d
	xi = xx(:,ii);
	new = -sin(2.*xi) .* (sin(ii.*xi.^2./(pi))).^(2.*m);
	sum  = sum + new;
end

y = -sum- 0.2.*xx(:,1);
end