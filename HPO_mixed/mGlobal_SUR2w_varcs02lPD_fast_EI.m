function [x_star, model] = mGlobal_SUR2w_varcs02lPD_fast_EI(model)
yd = size(model.init_value, 2);
canddata = model.candidate;
[y, s, model.MMRGP] = predict_resp(model.MMRGP, model.candidate);
mu = y(:, 1);
var1 = s(1:yd:end, 1);
sigma = sqrt(max(var1, 0));
f_best = max(model.now_value(:, 1));
safe_sigma = max(sigma, 1e-12);
Z = (mu - f_best) ./ safe_sigma;
PHI = 0.5 * (1 + erf(Z / sqrt(2)));
php = exp(-0.5 * Z.^2) / sqrt(2*pi);
EI = (mu - f_best) .* PHI + sigma .* php;
EI(sigma < 1e-12) = 0;
occ = ismember(model.candidate, model.now_x, 'rows');
EI(occ) = 0;
assert(sum(~occ) > 0, 'EI:候选池已全部被采样, 不应发生(池 2000 >> 预算 70)');
[~, index] = max(EI);
x_star = model.candidate(index, :);
model.M_ind(size(model.now_value, 1) - size(model.init_value, 1) + 1) = 0;
model.ei_last_mu = mu;
model.ei_last_sigma = sigma;
model.ei_last_ei = EI;
model.ei_last_fbest = f_best;
model.ei_last_index = index;
model.ei_last_occ_n = sum(occ);
model.candidate = canddata;
end
