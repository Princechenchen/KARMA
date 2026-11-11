function [y] =  Synthetic_2d(xx)
%【-4,4】
x1=xx(:,1);
x2=xx(:,2);
y=x1.*x2.*exp(-x1.^2-x2.^2);
end