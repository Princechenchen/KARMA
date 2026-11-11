function [y] = detpep10curv_3d(x1,x2,x3)
%[0,0.5]


term1 = 4 .* (x1 - 2 + 8.*x2 - 8.*x2.^2);
term2 = (4.*x2).^2;
term3 = 16 .* sqrt(x3+1) .* (2.*x1-1);

y = 0.6.*term2 + sin(term3+ term1);

end




