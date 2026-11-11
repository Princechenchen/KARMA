function f = himmelblau_2d(x)
% Himmelblau function
% 
% Input:
% x - a vector of length 2.
% 
% Output:
% f - the value of the Himmelblau function at x.
%[-6,6]
f = (x(:,1).^2 + x(:,2) - 11).^2 + (x(:,1) + x(:,2).^2 - 7).^2;
end