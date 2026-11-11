function [out]=ackley_6d3c(x)
%[0,1]
% dimension is # of columns of input, x1, x2, ..., xn
x1=ackley_6d(x);x2=ackley_6d3(x);
out = x1.*x2+x1+x2;
 out = out>=35;
 return