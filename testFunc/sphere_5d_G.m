function [y,latentb] = sphere_5d_G(xx)
ydata = sphere_5d(xx);
latentb = log(ydata);
bc = 1 ./ (1 + exp(-2*latentb+7.5)); 
bc(bc>0.5)=1; bc(bc<=0.5)=0; 
y1 = 2*latentb-1;
y2 = 2*latentb.^2;
y = [bc,y1,y2];
% yR = corrcoef(y);