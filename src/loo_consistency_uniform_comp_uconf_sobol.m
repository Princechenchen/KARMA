function [cons_vec, info] = loo_consistency_uniform_comp_uconf_sobol(model, opts)
if nargin < 2, opts = struct(); end
if ~isfield(opts, 'n_grid'), opts.n_grid = 51; end
if ~isfield(model,'pi_comp_fns') || numel(model.pi_comp_fns) < 2
    error('loo_consistency_uniform_comp_uconf_sobol: model.pi_comp_fns 需为 {π_hg, π_bnd} 二元 cell');
end
fn_hg = model.pi_comp_fns{1};  fn_bnd = model.pi_comp_fns{2};
X = model.now_x; Y = model.now_value;
ClassPos = model.ClassPos; dm = model.dim;
cont_cols = find(ClassPos == 1);
cls_cols  = find(ClassPos == 2);
n = size(X,1); yd = size(Y,2);
if isfield(opts,'ref_set') && ~isempty(opts.ref_set)
    U = opts.ref_set;
    if size(U,2) ~= dm
        error('loo_consistency_uniform_comp_uconf_sobol: ref_set 维度 %d ≠ dim %d', size(U,2), dm);
    end
    nu = size(U,1);
else
    g = linspace(0, 1, opts.n_grid);
    C = cell(1, dm);
    [C{:}] = ndgrid(g);
    U = cell2mat(cellfun(@(c) c(:), C, 'UniformOutput', false));
    nu = size(U,1);
end
Ystd_c = std(Y(:,cont_cols),1);  Ystd_c(Ystd_c < eps) = 1;
zchol = poschol(model.MMRGP.z_mn);
invz  = zchol \ (zchol' \ eye(size(zchol)));
Y_centered = model.MMRGP.ymn - model.MMRGP.fn * model.MMRGP.beta;
alpha_vec = invz * Y_centered(:);
e_cont = nan(n,1);
for i = 1:n
    blk = (i-1)*yd + (1:yd);
    V_ii = invz(blk, blk);
    V_ii = (V_ii + V_ii') / 2;
    r_blk = V_ii \ alpha_vec(blk);
    e_cont(i) = mean((r_blk(cont_cols)' ./ Ystd_c).^2);
end
D = dist(U, X');
W = 1 ./ (D.^2 + 1e-6);
eU_cont = (W * e_cont) ./ sum(W, 2);
uconf = zeros(nu,1);
if ~isempty(cls_cols)
    try
        j = cls_cols(1);
        [~, s_pred, pm] = predict_resp(model.MMRGP, U);
        fmu = pm.fmu(:);
        fs2 = s_pred((j:yd:yd*nu)', j);  fs2 = fs2(:);
        p_hat = exp(logphi(fmu ./ sqrt(1 + max(fs2,0))));
        uconf = 4*p_hat.*(1-p_hat);
    catch
        uconf = zeros(nu,1);
    end
end
pi_hg  = fn_hg(U);   pi_hg  = pi_hg(:);
pi_bnd = fn_bnd(U);  pi_bnd = pi_bnd(:);
cons_hg = 0;
if std(pi_hg) >= eps && std(eU_cont) >= eps
    R = corrcoef(pi_hg, eU_cont);  cons_hg = R(1,2);
    if isnan(cons_hg), cons_hg = 0; end
end
cons_bnd = 0;
if std(pi_bnd) >= eps && std(uconf) >= eps
    R = corrcoef(pi_bnd, uconf);  cons_bnd = R(1,2);
    if isnan(cons_bnd), cons_bnd = 0; end
end
cons_vec = [cons_hg, cons_bnd];
info = struct('n_tr', n, 'n_grid', opts.n_grid, 'nu', nu, ...
              'mode', 'uniform-idw-comp-uconf-sobol');
end
