function [y] = wingweight(xx)
y1 = wingweight1(xx);
y2 = wingweight2(xx);
yc = y1<=250 & y2<=50;

y = [y1,y2,yc];
% yR = corrcoef(y);