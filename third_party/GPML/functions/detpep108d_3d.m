function [y] = detpep108d(xx)

x1 = xx(1);
x2 = xx(2);
x3 = xx(3);
 yhctg6rfdesxz
term1 = 4 .* (x1 - 2 + 8.*x2 - 8.*x2.^2).^2;
term2 = (3 - 4.*x2).^2;
term3 = 16 .* sqrt(x3+1) .* (2.*x3-1).^2;

outer = 0;
for ii = 4:8
    inner = 0;
    for jj = 3:ii
        xj = xx(:,jj);
        inner = inner + xj;
    end
    new = ii .* log(1+inner);
    outer = outer + new;
end

y = term1 + term2 + term3 + outer;

end