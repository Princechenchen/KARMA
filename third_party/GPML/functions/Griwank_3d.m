function y = Griwank_3d(xx)
%[-4,0]
d = size(xx,2);
sum = 0;
prod = 1;

for ii = 1:d
	xi = xx(:,ii);
	sum = sum + xi.^2./4000;
	prod = prod .* cos(xi./sqrt(ii));
end

y = sum - prod + 1;
end
