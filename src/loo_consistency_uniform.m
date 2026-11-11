function [consistency, info] = loo_consistency_uniform(model, opts)
if nargin < 2, opts = struct(); end
if ~isfield(opts, 'n_grid'), opts.n_grid = 51; end
dm = model.dim;
g = linspace(0, 1, opts.n_grid);
C = cell(1, dm);
[C{:}] = ndgrid(g);
U = cell2mat(cellfun(@(c) c(:), C, 'UniformOutput', false));
nu = size(U,1);
[~, li] = loo_consistency_fast(model, struct('loo_max', Inf));
Xtr = model.now_x(li.idx, :);
e_tr = li.e;
D = dist(U, Xtr');
W = 1 ./ (D.^2 + 1e-6);
eU = (W * e_tr) ./ sum(W, 2);
pi_at = model.pi_phys_fn(U);
if nu < 3 || numel(e_tr) < 3 || std(eU) < eps || std(pi_at(:)) < eps
    consistency = 0;
else
    R = corrcoef(pi_at(:), eU);
    consistency = R(1,2);
    if isnan(consistency), consistency = 0; end
end
info = struct('n_grid', opts.n_grid, 'n_used', nu, 'n_tr', numel(e_tr), ...
    'mode', 'uniform-idw', 'pi_at', pi_at(:), 'eU', eU);
end
