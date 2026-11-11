function [ref, cache_out] = pi24_referee_step(model, cache, opts)
if nargin < 2 || isempty(cache) || ~isfield(cache, 'step')
    cache = struct('step', {{}}, 'E', {{}}, 'rho', {{}});
end
if nargin < 3, opts = struct(); end
Delta = 2;    if isfield(opts, 'Delta')  && ~isempty(opts.Delta),  Delta  = opts.Delta;  end
B      = 200; if isfield(opts, 'B')      && ~isempty(opts.B),      B      = opts.B;      end
knn_k  = 5;   if isfield(opts, 'knn_k')  && ~isempty(opts.knn_k),  knn_k  = opts.knn_k;  end
sob_n  = 4000; if isfield(opts, 'sobol_n') && ~isempty(opts.sobol_n), sob_n = opts.sobol_n; end
pri = [];
if isfield(opts, 'pri') && ~isempty(opts.pri), pri = opts.pri; end
X = model.now_x; Y = model.now_value;
ClassPos = model.ClassPos; dm = model.dim;
cont_cols = find(ClassPos == 1); cls_cols = find(ClassPos == 2);
n = size(X, 1); yd = size(Y, 2); nRC = numel(cont_cols);
sob = sobolset(dm, 'Skip', 4096, 'Leap', 1);
rng_guard = rng;  rng(2026, 'twister');
U = net(sob, sob_n);
rng(rng_guard);
nu = size(U, 1);
zchol = poschol(model.MMRGP.z_mn);
invz = zchol \ (zchol' \ eye(size(zchol)));
Y_centered = model.MMRGP.ymn - model.MMRGP.fn * model.MMRGP.beta;
alpha_vec = invz * Y_centered(:);
Ystd_c = std(Y(:, cont_cols), 1);  Ystd_c(Ystd_c < eps) = 1;
r2 = nan(n, nRC);
for i = 1:n
    blk = (i-1)*yd + (1:yd);
    V_ii = invz(blk, blk);  V_ii = (V_ii + V_ii') / 2;
    r_blk = V_ii \ alpha_vec(blk);
    r2(i, :) = (r_blk(cont_cols)' ./ Ystd_c).^2;
end
D = dist(U, X');
W = 1 ./ (D.^2 + 1e-6);
Ws = sum(W, 2);
r2U = zeros(nu, nRC);
for k = 1:nRC
    r2U(:, k) = (W * r2(:, k)) ./ Ws;
end
uconf = zeros(nu, 1);
if ~isempty(cls_cols)
    try
        j = cls_cols(1);
        [~, s_pred, pm] = predict_resp(model.MMRGP, U);
        fmu = pm.fmu(:);
        fs2 = s_pred((j:yd:yd*nu)', j);  fs2 = fs2(:);
        p_hat = exp(logphi(fmu ./ sqrt(1 + max(fs2, 0))));
        uconf = 4 * p_hat .* (1 - p_hat);
    catch
        uconf = zeros(nu, 1);
    end
end
E = zeros(nu, 1);
for k = 1:nRC
    mk = median(r2U(:, k));  if mk < eps, mk = 1; end
    E = E + r2U(:, k) / mk;
end
mu_ = median(uconf);  if mu_ < eps, mu_ = 1; end
E = (E + uconf / mu_) / (nRC + 1);
kk = max(1, min(knn_k, n));
Dk = sort(pdist2(U, X), 2);
dk = Dk(:, kk) + 1e-12;
Vd = pi^(dm/2) / gamma(dm/2 + 1);
rho = kk ./ (n * Vd * dk.^dm);
n_step = n - model.n_init + 1;
cache_out = cache;
cache_out.step{end+1, 1} = n_step;
cache_out.E{end+1, 1} = E;
cache_out.rho{end+1, 1} = rho;
keep = numel(cache_out.step);
if keep > Delta + 1
    idx = (keep - Delta):keep;
    cache_out.step = cache_out.step(idx);
    cache_out.E = cache_out.E(idx);
    cache_out.rho = cache_out.rho(idx);
end
ref = struct('n_step', n_step, 'active', false, 'w', [0.5 0.5], ...
    'w_pois', [NaN NaN], 'Lam', [NaN NaN], 'Lam_pois', [NaN NaN], 'Lam_tilde', []);
i_prev = find([cache_out.step{:}] == n_step - Delta, 1);
if isempty(i_prev) || isempty(pri)
    return
end
ref.active = true;
E_prev = cache_out.E{i_prev};  rho_prev = cache_out.rho{i_prev};
epsn = 0.1 * median(E);  if epsn < eps, epsn = 1e-12; end
g = ((E_prev + epsn) ./ (E + epsn)) .* (rho_prev ./ rho);
fn_hg = model.pi_comp_fns{1};  fn_bnd = model.pi_comp_fns{2};
pi_hg_U = fn_hg(U);  pi_hg_U = pi_hg_U(:);
pi_bnd_U = fn_bnd(U); pi_bnd_U = pi_bnd_U(:);
lam_of = @(wf) (sum(wf .* g) / sum(wf)) / (sum((1 - wf) .* g) / sum(1 - wf));
ref.Lam = [lam_of(pi_hg_U), lam_of(pi_bnd_U)];
wd = pri.win_dims;  nw = numel(wd);
sU = U * (pri.w / sum(pri.w))';
rng_guard = rng;  rng(20260109, 'twister');
u_shift = rand(B, nw) - 0.5;
u_q = 0.1 + 0.8 * rand(B, 1);
rng(rng_guard);
Lam_tilde = nan(B, 2);
for b = 1:B
    cb = mod(pri.win_c + u_shift(b, :), 1);
    ph = zeros(nu, 1);
    for t = 1:nw
        ph = max(ph, max(0.01, exp(-0.5 * ((U(:, wd(t)) - cb(t)) / pri.win_s).^2)));
    end
    s0b = quantile(sU, u_q(b));
    pb = max(0.01, exp(-0.5 * ((sU - s0b) / pri.sigma_b).^2));
    Lam_tilde(b, 1) = lam_of(ph);
    Lam_tilde(b, 2) = lam_of(pb);
end
ph_p = zeros(nu, 1);
for t = 1:nw
    ph_p = max(ph_p, max(0.01, exp(-0.5 * ((U(:, wd(t)) - 0.90) / pri.win_s).^2)));
end
s0p = quantile(sU, 0.90);
pb_p = max(0.01, exp(-0.5 * ((sU - s0p) / pri.sigma_b).^2));
ref.Lam_pois = [lam_of(ph_p), lam_of(pb_p)];
ref.Lam_tilde = Lam_tilde;
ref.w = (0.5 + [sum(Lam_tilde(:, 1) < ref.Lam(1)), sum(Lam_tilde(:, 2) < ref.Lam(2))]) / (B + 1);
ref.w_pois = (0.5 + [sum(Lam_tilde(:, 1) < ref.Lam_pois(1)), sum(Lam_tilde(:, 2) < ref.Lam_pois(2))]) / (B + 1);
end
