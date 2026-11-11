function hypd = D_likelihood(x,y,alpha,L,K,hyp,dist_matrix)
% -1/2*(y-muy)'*alpha-sum(diag(log(L)))-numel(y)/2*log(2*pi); 
% sf = hyp.cov(2); ell= hyp.cov(1);sn = hyp.lik;
sn = hyp(1); sf = hyp(2); ell= hyp(3:end);
invK = L'\(L\eye(size(x,1)));
aa = alpha*alpha';

derK_ell = sf.^2.*dist_matrix.*exp(dist_matrix./(-2*ell^2))./ell^3;
d_ell = 1/2*trace((aa-invK)*derK_ell);
hypd.cov(1) = d_ell;

derK_sf = 2*sf.*exp(dist_matrix./(-2*ell^2));
d_sf = 1/2*trace((aa-invK)*derK_sf);
hypd.cov(2) = d_sf;

derK_sn = 2*eye(size(x,1))*sn;
d_lik = 1/2*trace((aa-invK)*derK_sn);
hypd.lik = d_lik;
end