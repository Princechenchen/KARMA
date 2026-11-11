function [consistency, info] = loo_consistency_fast(model, opts)
if nargin < 2, opts = struct(); end
if ~isfield(opts,'loo_max'), opts.loo_max = Inf; end
X = model.now_x; Y = model.now_value;
ClassPos = model.ClassPos; dm = model.dim;
cont_cols = find(ClassPos == 1);
n = size(X,1);
yd = size(Y,2);
idx = (1:n)';
if n > opts.loo_max
    idx = round(linspace(1, n, opts.loo_max));
    idx = unique(idx)';
end
m = numel(idx);
Ystd = std(Y(:,cont_cols),1);
Ystd(Ystd < eps) = 1;
zchol = poschol(model.MMRGP.z_mn);
invz  = zchol \ (zchol' \ eye(size(zchol)));
Y_centered = model.MMRGP.ymn - model.MMRGP.fn * model.MMRGP.beta;
alpha_vec = invz * Y_centered(:);
e = nan(m,1);
resid_std = nan(m, numel(cont_cols));
for k = 1:m
    i = idx(k);
    blk = (i-1)*yd + (1:yd);
    V_ii = invz(blk, blk);
    V_ii = (V_ii + V_ii') / 2;
    r_blk = V_ii \ alpha_vec(blk);
    resid_std(k, :) = r_blk(cont_cols)' ./ Ystd;
    e(k) = mean(resid_std(k, :).^2);
end
good = ~isnan(e);
e_good = e(good);
pi_at = model.pi_phys_fn(X(idx(good),:));
if numel(e_good) < 3 || std(e_good) < eps || std(pi_at) < eps
    consistency = 0;
else
    R = corrcoef(pi_at, e_good);
    consistency = R(1,2);
    if isnan(consistency), consistency = 0; end
end
info = struct('idx', idx(good), 'e', e_good, 'pi_at', pi_at, 'n_used', sum(good), ...
    'mode', 'virtual', 'resid_std', resid_std(good,:));
end
