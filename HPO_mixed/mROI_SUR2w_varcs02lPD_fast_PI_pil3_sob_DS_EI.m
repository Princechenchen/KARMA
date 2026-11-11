function [x_star, model] = mROI_SUR2w_varcs02lPD_fast_PI_pil3_sob_DS_EI(model)
ns = size(model.candidate,1); yd = size(model.init_value,2); nts = size(model.testsample,1);
canddata = model.candidate;
testsample = model.testsample;
dim = model.dim;
lower_bounds=zeros(1,dim);
upper_bounds=ones(1,dim);
bounds = [lower_bounds; upper_bounds]';
num_samples = 50;
iterations1 = 200;
RoI_num=floor(dim*2);
iterations2 = round(iterations1/RoI_num);
elitism_size = 10;
model_num=1;
window_min=0.35;
window_max=0.85;
window_size_bounds = zeros(dim, 2);
for d = 1:dim
    window_size_bounds(d, :) = [(upper_bounds(d) - lower_bounds(d)) * window_min, ...
        (upper_bounds(d) - lower_bounds(d)) * window_max];
end
roi_cache = prepare_roi_predict_cache(model.MMRGP);
fast_options = struct('batch_prediction', true, 'batch_archive', true, ...
    'reuse_sampling_template', true, 'vectorized_roi_metrics', true);
if isfield(model,'pi_comp_adaptive') && model.pi_comp_adaptive
    eta = 0.5;  if isfield(model,'pi_comp_eta')       && ~isempty(model.pi_comp_eta),       eta  = model.pi_comp_eta;       end
    amin = 0.1; if isfield(model,'pi_comp_alpha_min') && ~isempty(model.pi_comp_alpha_min), amin = model.pi_comp_alpha_min; end
    a_prev = [1 1]; if isfield(model,'pi_comp_alpha') && ~isempty(model.pi_comp_alpha), a_prev = model.pi_comp_alpha; end
    uopts = struct('n_grid', 51);
    if isfield(model,'pi_unif_grid') && ~isempty(model.pi_unif_grid), uopts.n_grid = model.pi_unif_grid; end
    nu_sob = 8192; if isfield(model,'pi_unif_sobol_n') && ~isempty(model.pi_unif_sobol_n), nu_sob = model.pi_unif_sobol_n; end
    if model.dim >= 8 && ~isempty(which('sobolset'))
        sob = sobolset(model.dim, 'Skip', 4096, 'Leap', 1);
        rng_sob_guard = rng;  rng(2026, 'twister');
        uopts.ref_set = net(sob, nu_sob);
        rng(rng_sob_guard);
    end
    [cons_vec, ~] = loo_consistency_uniform_comp_uconf_sobol(model, uopts);
    a_new = max(amin, eta*a_prev + (1-eta)*cons_vec);
    model.pi_comp_alpha = a_new;
    if ~isfield(model,'pi_comp_cons_hist')  || isempty(model.pi_comp_cons_hist),  model.pi_comp_cons_hist  = zeros(0,2); end
    if ~isfield(model,'pi_comp_alpha_hist') || isempty(model.pi_comp_alpha_hist), model.pi_comp_alpha_hist = zeros(0,2); end
    model.pi_comp_cons_hist(end+1,:)  = cons_vec;
    model.pi_comp_alpha_hist(end+1,:) = a_new;
    fprintf('[_PI-pil3sobDS] step %d : α_hg %.3f→%.3f (cons %.3f) | α_bnd %.3f→%.3f (cons %.3f)\n', ...
        size(model.now_x,1), a_prev(1), a_new(1), cons_vec(1), a_prev(2), a_new(2), cons_vec(2));
end
a_use = [1 1]; if isfield(model,'pi_comp_alpha') && ~isempty(model.pi_comp_alpha), a_use = model.pi_comp_alpha; end
fn_hg = model.pi_comp_fns{1};  fn_bnd = model.pi_comp_fns{2};
fused_fn = @(X) max(fn_hg(X).^a_use(1), fn_bnd(X).^a_use(2));
knn_k = 5;  L_anneal = 3;  tau_drop = 0.25;
if isfield(model,'pi3_anneal')
    if isfield(model.pi3_anneal,'k'),        knn_k   = model.pi3_anneal.k;         end
    if isfield(model.pi3_anneal,'L'),        L_anneal= model.pi3_anneal.L;         end
    if isfield(model.pi3_anneal,'tau_drop'), tau_drop= model.pi3_anneal.tau_drop;  end
end
Xv = model.now_x;
pi_eff_fn = @(X) fused_fn(X) .* prior_anneal_factor(X, Xv, knn_k, L_anneal);
alpha_eff = max(a_use);
pi_on = alpha_eff >= tau_drop;
model.pi_phys_fn = pi_eff_fn;
model.pi_alpha   = 1;
model.pi3_active = pi_on;
model.pi3_w_phi  = alpha_eff;
if ~isfield(model,'pi3_hist') || isempty(model.pi3_hist), model.pi3_hist = zeros(0,4); end
model.pi3_hist(end+1,:) = [a_use, alpha_eff, double(pi_on)];
fprintf('[_PI-pil3] step %d : α_eff=%.3f gate=%d w_phi=%.3f | 退火 k=%d L=%.1f τ=%.2f\n', ...
    size(model.now_x,1), alpha_eff, pi_on, alpha_eff, knn_k, L_anneal, tau_drop);
occupied_table = mopso_adaptive_elitism_crowding_PI3(dim, bounds, window_size_bounds, ...
    model, num_samples, iterations1, iterations2, num_samples, elitism_size, ...
    model_num, roi_cache, fast_options);
model.last_roi_table = occupied_table;
occupied = table2array(occupied_table); occupied = occupied(:,1:2*dim);
pool = canddata;
in_any = false(size(pool,1),1);
for i = 1:size(occupied,1)
    upb = occupied(i,2:2:end); dob = occupied(i,1:2:end);
    in_any = in_any | all(pool<=upb & pool>=dob, 2);
end
in_idx = find(in_any); out_idx = find(~in_any);
globalnum = floor(size(pool,1)*0.1);
if numel(out_idx) < globalnum
    sel = [in_idx; out_idx];
else
    rp = randperm(numel(out_idx), globalnum);
    sel = [in_idx; out_idx(rp(:))];
end
idcand = sel;
model.candidate = pool(sel,:);
ns = size(model.candidate,1);
ns = size(model.candidate,1); nts = size(model.testsample,1);
[y, s, model.MMRGP] = predict_resp(model.MMRGP, model.candidate);
mu = y(:, 1);
var1 = s(1:yd:end, 1);
sigma = sqrt(max(var1, 0));
f_best = max(model.now_value(:, 1));
safe_sigma = max(sigma, 1e-12);
Z = (mu - f_best) ./ safe_sigma;
PHI = 0.5 * (1 + erf(Z / sqrt(2)));
php = exp(-0.5 * Z.^2) / sqrt(2*pi);
metric = (mu - f_best) .* PHI + sigma .* php;
metric(sigma < 1e-12) = 0;
occ = ismember(model.candidate, model.now_x, 'rows');
metric(occ) = 0;
assert(sum(~occ) > 0, 'EI:区域内候选已全部被采样');
model.ei_roi_last_fbest = f_best; model.ei_roi_last_occ_n = sum(occ); model.ei_roi_last_n = ns;
[bestvalue,index] = max(metric);
x_star = model.candidate(index,:);
model.M_ind(size(model.now_value,1)-size(model.init_value,1)+1)=idcand(index);
model.candidate = canddata;
model.testsample = testsample;
end
