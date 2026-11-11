function y = sphere_5d(xx)
%[-5.12,5.12]
d = size(xx,2);
sum = 0;
for ii = 1:d
	xi = xx(:,ii);
	sum = sum + xi.^2;
end

y = sum;
end