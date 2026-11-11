function [y]=ackley(xx)
y1=ackley_6d(xx);
y2=ackley_6d3(xx);
y3=ackley_6d3c(xx);
y=[y1,y2,y3];
 return