function K= covSEfunc(hyp, x, z,corrfunc)


n=size(x,1);
m=size(z,1);

% for i=1:n
%     for j=1:m
%         K(i,j)=exp(-sum( hyp.*((x(i,:)-z(j,:)).^2) ));
%     end
% end
% % K= A+A'-diag(diag(A));
z = repmat(z,n,1); 
x = repelem(x,m,1);
d = (x-z);
switch true
    case corrfunc=="corrgauss"
        K = exp(-sum( hyp.*(d.^2),2));
        K = reshape(K,m,n); K=K';
    case corrfunc=="correxpg"
        pow = hyp(end);   tt = -hyp(1:end-1);
        td = abs(d).^pow .* tt;
        K = exp(sum(td,2));
        K = reshape(K,m,n); K=K';
    case corrfunc=="correxp"
        td = -hyp.*abs(d);
        K = exp(sum(td,2));
        K = reshape(K,m,n); K=K';
    case corrfunc=="corrcubic"
        td = min(hyp.*abs(d), 1);
        ss = 1 - td.^2 .* (3 - 2*td);
        K = prod(ss, 2);
        K = reshape(K,m,n); K=K';
    case corrfunc=="corrlin"
        td = max(1 - hyp.*abs(d), 0);
        K = prod(td, 2);
        K = reshape(K,m,n); K=K';
    case corrfunc=="corrspherical"
        td = min(hyp.*abs(d), 1);
        ss = 1 - td .* (1.5 - .5*td.^2);
        K = prod(ss, 2);
        K = reshape(K,m,n); K=K';
    case corrfunc=="corrspline"  
        dn = size(x,2);
        mdn = m*dn*n;   ss = zeros(mdn,1);
        xi = reshape(hyp.*abs(d), mdn,1);
        % Contributions to first and second part of spline
        i1 = find(xi <= 0.2);
        i2 = find(0.2 < xi & xi < 1);
        if  ~isempty(i1)
          ss(i1) = 1 - xi(i1).^2 .* (15  - 30*xi(i1));
        end
        if  ~isempty(i2)
          ss(i2) = 1.25 * (1 - xi(i2)).^3;
        end
        % Values of correlation
        ss = reshape(ss,m*n,dn);
        K = prod(ss, 2);
        K = reshape(K,m,n); K=K';
end