function y = Zakharov_4d3(xx)
% [-10,10]
d = size(xx,2);
sum1 = 0;
sum2 = 0;

for ii = 1:d
	xi = xx(:,ii);
	sum1 = sum1 + xi.^2;
	sum2 = sum2 + 0.5.*ii.*xi;
end

y = sum1 + sum2.^3 + sum2.^5;
end
