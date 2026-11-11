function [consistency, info] = loo_consistency(model, opts)
if nargin < 2, opts = struct(); end
if ~isfield(opts,'loo_max'), opts.loo_max = 30; end
X = model.now_x; Y = model.now_value;
ClassPos = model.ClassPos; dm = model.dim;
cont_cols = find(ClassPos == 1);
n = size(X,1);
idx = (1:n)';
if n > opts.loo_max
    idx = round(linspace(1, n, opts.loo_max));
    idx = unique(idx)';
end
m = numel(idx);
Ystd = std(Y(:,cont_cols),1);
Ystd(Ystd < eps) = 1;
e = nan(m,1);
for k = 1:m
    i = idx(k);
    keep = true(n,1); keep(i) = false;
    try
        mloo = MMRGP0(Y(keep,:), X(keep,:), ClassPos, dm);
        yhat = predict_resp(mloo, X(i,:));
        resid = (yhat(1,cont_cols) - Y(i,cont_cols)) ./ Ystd;
        e(k) = mean(resid.^2);
    catch
        e(k) = nan;
    end
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
info = struct('idx', idx(good), 'e', e_good, 'pi_at', pi_at, 'n_used', sum(good));
end
