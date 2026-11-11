function y = Colville_4d(xx)
% xx=xx.*2.5;
x1=xx(:,1);
x2=xx(:,2);
x3=xx(:,3);
x4=xx(:,4);
%【-2,2】
if any([x1 < 0, x2 < 0])
    term1 = -100 .* ((x1-x2).^3).*sin(x3);
term2 = (x1-4).^2;
else
term1 = 100 .* (x1.^2-x2).^2;
term2 = (x1-1).^2;
end
term3 = (x3-1).^2;
term4 = 90 .* (x3.^2-x4).^2;
term5 = 10.1 .* ((x2-1).^2 + (x4-1).^2);
term6 = 19.8.*(x2-1).*(x4-1);

y = term1 + term2 + term3 + term4 + term5 + term6;
end

