function y = Styblinski_Tang_5d(xx)
%[-5,5]
d = size(xx,2);
sum = 0;
for ii = 1:d
	xi = xx(:,ii);
	new = xi.^4 - 16.*xi.^2 + 5.*xi;
	sum = sum + new;
end

y = sum./2;

end
