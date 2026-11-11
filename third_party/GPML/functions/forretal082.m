function [y] = forretal082(x)
% x=x.*1.05;
fact1 = (6.*x - 2).^2;
fact2 = sin(12.*x - 4);
fact3=1.5.*(x+2.5);
y = sqrt(fact1 .* fact2+10).*fact3;
end