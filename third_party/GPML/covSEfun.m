%协方差函数（平方指数协方差函数）

function [K,dist_matrix] = covSEfun(hyp,x1,x2)
    sf = hyp.cov(2); ell= hyp.cov(1);
    m=size(x1,1);n=size(x2,1);
    dist_matrix=zeros(m,n);%距离矩阵
    for i=1:m
        for j=1:n
            dist_matrix(i,j)=sum((x1(i,:)-x2(j,:)).^2);    
            if (i==j)
            K(i,j)=sf^2.*exp(-dist_matrix(i,j)/2/(ell^2));
            else
            K(i,j)=sf^2.*exp(-dist_matrix(i,j)/2/(ell^2));    
            end
        end
    end
    
    
%     mu = (m/(n+m))*mean(x1,1) + (n/(n+m))*mean(x2,1); 
%     x1 = x1-mu;
%     x2 = x2-mu;
%     Ax1 = x1./ell^2; sAx1 = sum(x1.*Ax1,2);
%     Ax2 = x2./ell^2; sAx2 = sum(x2.*Ax2,2);
%     dist_matix = sAx1-2*x1*Ax2'+sAx2';
% %     D2 = max(bsxfun(@plus,sax,bsxfun(@minus,saz',2*x*Az')),0);     % computation
%     K = sf^2.*exp(dist_matix/2);
end