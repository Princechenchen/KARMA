function [y] = Rosenbrock_2d(xx)
%[-2.5,2.5]
d = 2;
sum = 0;
for ii = 1:(d-1)
	xi = xx(:,ii);
	xnext = xx(:,ii+1);
	new = 100.*(xnext-xi.^2).^2 + (xi-1).^2;
    if xx(2) > 1.5
       new = new + 700.*xx(2).*xx(1); 
    end
	sum = sum + new;
end

y = sum;
end
