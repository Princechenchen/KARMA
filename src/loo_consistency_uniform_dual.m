function [consistency, info] = loo_consistency_uniform_dual(model, opts)
if nargin < 2, opts = struct(); end
if ~isfield(opts, 'n_grid'), opts.n_grid = 51; end
if ~isfield(opts, 'w_cont'), opts.w_cont = 0.5; end
if ~isfield(opts, 'w_cls'),  opts.w_cls  = 0.5; end
X = model.now_x; Y = model.now_value;
ClassPos = model.ClassPos; dm = model.dim;
cont_cols = find(ClassPos == 1);
cls_cols  = find(ClassPos == 2);
n = size(X,1); yd = size(Y,2);
g = linspace(0, 1, opts.n_grid);
C = cell(1, dm);
[C{:}] = ndgrid(g);
U = cell2mat(cellfun(@(c) c(:), C, 'UniformOutput', false));
nu = size(U,1);
Ystd_c  = std(Y(:,cont_cols),1);  Ystd_c(Ystd_c < eps) = 1;
if ~isempty(cls_cols)
    Ystd_s = std(Y(:,cls_cols),1);  Ystd_s(Ystd_s < eps) = 1;
end
zchol = poschol(model.MMRGP.z_mn);
invz  = zchol \ (zchol' \ eye(size(zchol)));
Y_centered = model.MMRGP.ymn - model.MMRGP.fn * model.MMRGP.beta;
alpha_vec = invz * Y_centered(:);
e_cont = nan(n,1);  e_cls = nan(n,1);
for i = 1:n
    blk = (i-1)*yd + (1:yd);
    V_ii = invz(blk, blk);
    V_ii = (V_ii + V_ii') / 2;
    r_blk = V_ii \ alpha_vec(blk);
    e_cont(i) = mean((r_blk(cont_cols)' ./ Ystd_c).^2);
    if ~isempty(cls_cols)
        e_cls(i) = mean((r_blk(cls_cols)' ./ Ystd_s).^2);
    end
end
D = dist(U, X');
W = 1 ./ (D.^2 + 1e-6);
eU_cont = (W * e_cont) ./ sum(W, 2);
if ~isempty(cls_cols)
    eU_cls = (W * e_cls) ./ sum(W, 2);
else
    eU_cls = zeros(nu,1);
end
pi_at = model.pi_phys_fn(U);  pi_at = pi_at(:);
ok_pi = std(pi_at) >= eps;
cons_cont = 0; cons_cls = 0; has_c = false; has_s = false;
if ok_pi && std(eU_cont) >= eps
    R = corrcoef(pi_at, eU_cont);  cons_cont = R(1,2);
    has_c = ~isnan(cons_cont);  if ~has_c, cons_cont = 0; end
end
if ok_pi && ~isempty(cls_cols) && std(eU_cls) >= eps
    R = corrcoef(pi_at, eU_cls);   cons_cls  = R(1,2);
    has_s = ~isnan(cons_cls);   if ~has_s, cons_cls = 0; end
end
w_sum = opts.w_cont*has_c + opts.w_cls*has_s;
if w_sum < eps
    consistency = 0;  wc = 0; ws = 0;
else
    wc = opts.w_cont*has_c / w_sum;  ws = opts.w_cls*has_s / w_sum;
    consistency = wc*cons_cont + ws*cons_cls;
end
info = struct('cons_cont', cons_cont, 'cons_cls', cons_cls, ...
    'w_cont_eff', wc, 'w_cls_eff', ws, 'n_tr', n, 'n_grid', opts.n_grid, ...
    'mode', 'uniform-idw-dual');
end
