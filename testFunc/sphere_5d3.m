function y = sphere_5d3(xx)
%[-5.12,5.12]
% xx = mapminmax(xx',-5.12,5.12); xx=xx';
xx = xx*5.12*2-5.12;  %[0,1]->[-5.12,5.12]
d = size(xx,2);
sum = 0;
for ii = 1:d
	xi = xx(:,ii)+0.2;
	sum = sum + xi.^2;
end

y = sum;
end