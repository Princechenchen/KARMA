function [likeval,alpha] = fitfun_class(hypx,y,x,alpha)
%hypx-超参数(mean,sn,sf,ell) y-响应 x-因子
for i=1:size(hypx,1)
[K,~]= covSEfun2(hypx(i,2:end),x,x);   %协方差矩阵
% alpha = zeros(size(y,1),1);
m=ones(size(y,1),1).*hypx(i,1);
alpha(:,i) = Nt_iter_ls(alpha(:,i), m,K,y);
f = K*alpha(:,i)+m;   yf = y.*f;                            % product latents and labels
[lp,dlp,d2lp] = logphi(yf); W = -d2lp; n = size(K,1); dlp = dlp.*y;   %logp(f|x,y)及导数
% symmetric B = I+sW*K*sW
sW = sqrt(W); 
L = chol(eye(n)+sW*sW'.*K,"lower");             % Cholesky factor of B
ldB2 = sum(log(diag(L)));                                    % log(det(B))/2
% compute marginal likelihood
likeval(i,1) = -1*(-alpha(:,i)'*(f-m)/2 + sum(lp) - ldB2);             %Laplace近似的似然函数，logq(y|X,θ) = -1/2f'inv(K)f+logp(y|f)-1/2log|B|
end
end