function [out]=Rastrigin_2d(in)
%[0,1]
 cos_in = cos(2*pi*in);
 out    = sum((in.^2-10*cos_in + 10), 2);
end