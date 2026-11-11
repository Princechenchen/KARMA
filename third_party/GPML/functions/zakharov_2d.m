function [y] = zakharov_2d(x1,x2)
xx=[x1,x2];
d = length(xx);
sum1 = 0;
sum2 = 0;
sum1 = sum1 + x1.^2;
sum2 = sum2 + 0.5.*1.*x1;
sum1 = sum1 + x2.^2;
sum2 = sum2 + 0.5.*2.*x2;
y = sum1 + sum2.^2 + sum2.^4;

end