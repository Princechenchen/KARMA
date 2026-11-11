function [y] = f10(xx)
xx=xx.*6-3;
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
