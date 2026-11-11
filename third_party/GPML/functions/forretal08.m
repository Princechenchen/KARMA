function [y] = forretal08(x)
% x=x.*1.05;
fact1 = (6.*x - 2).^2;
fact2 = sin(12.*x - 4);
y = fact1 .* fact2+10;
end