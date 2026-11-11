function [y,latentb] = Branin_G(xx)
ydata = Branin_2d3(xx);
latentb = log(ydata);
bc = 1 ./ (1 + exp(-latentb+4)); 
bc(bc>0.5)=1; bc(bc<=0.5)=0; 
y1 = 2*latentb-3;
y2 = latentb.^2;
y = [bc,y1,y2];
% yR = corrcoef(y);