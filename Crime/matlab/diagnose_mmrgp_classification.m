function diagnostics = diagnose_mmrgp_classification(train_x, train_y2, gpml_root)
% Diagnose the exact GPML Laplace system without changing the formal model.
addpath(gpml_root);
for d = {'cov','doc','inf','lik','mean','prior','util'}
    addpath(fullfile(gpml_root,d{1}));
end
for d = {{'util','minfunc'},{'util','minfunc','compiled'}}
    addpath(fullfile(gpml_root,d{1}{:}));
end
addpath(fullfile(gpml_root,'util','sparseinv'));
meanfunc = @meanConst;
covfunc = @covSEard;
likfunc = @likErf;
hyp = struct('mean',0,'cov',zeros(1,size(train_x,2)+1));
y = train_y2(:);
y(y ~= 1) = -1;
hyp = minimize(hyp,@gp,-40,@infLaplace,meanfunc,covfunc,likfunc,train_x,y);
hyp.lik = [];
K = covSEard(hyp.cov,train_x);
diagnostics = struct();
diagnostics.hyp_mean = hyp.mean;
diagnostics.hyp_cov = hyp.cov;
diagnostics.min_eig_K = min(eig((K+K')/2));
diagnostics.finite_hyp = all(isfinite([hyp.mean;hyp.cov(:)]));
diagnostics.finite_K = all(isfinite(K(:)));
diagnostics.n = length(y);
diagnostics.n_positive = sum(y==1);
scales = exp(-hyp.cov(1:end-1));
D2_direct = zeros(size(train_x,1));
for i = 1:size(train_x,1)
    delta = bsxfun(@minus,train_x,train_x(i,:));
    D2_direct(:,i) = sum(bsxfun(@times,delta.^2,scales.^2),2);
end
K_direct = exp(2*hyp.cov(end))*exp(-D2_direct/2);
diagnostics.min_eig_K_direct = min(eig((K_direct+K_direct')/2));
diagnostics.max_abs_K_difference = max(abs(K(:)-K_direct(:)));
try
    post = infLaplace(hyp,{meanfunc},{covfunc},{likfunc},train_x,y);
    m = meanConst(hyp.mean,train_x);
    f = K*post.alpha+m;
    [~,~,d2lp] = likErf([],y,f,[],'infLaplace');
    W = -d2lp;
    sW = sqrt(abs(W)).*sign(W);
    B = eye(size(K))+sW*sW'.*K;
    diagnostics.min_eig_B = min(eig((B+B')/2));
    diagnostics.min_W = min(W);
    diagnostics.max_W = max(W);
    diagnostics.max_abs_alpha = max(abs(post.alpha));
    diagnostics.max_abs_f = max(abs(f));
    diagnostics.finite_B = all(isfinite(B(:)));
    diagnostics.post_error = '';
catch err
    diagnostics.post_error = err.message;
end
end
