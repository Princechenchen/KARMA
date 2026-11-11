function IL_ns = calculate_entropy_fast(x, sampledY, erro)
% Vectorized equivalent of calculate_entropy for ROI information surface.
n = size(x, 1);
if n == 0
    IL_ns = 0.0;
    return;
end

x_star = best_point(x, sampledY);
y_min = min(sampledY, [], 1);
y_max = max(sampledY, [], 1);
y_range = y_max - y_min;
y_range(y_range == 0) = 1;
f = (sampledY - y_min) ./ y_range;

pair_counts = n - (1:n-1);
ii = repelem((1:n-1)', pair_counts(:));
pair_cells = arrayfun(@(i) (i+1:n)', 1:n-1, 'UniformOutput', false);
jj = vertcat(pair_cells{:});
if isempty(ii)
    IL_ns = 0.0;
    return;
end

response_diff = f(ii, :) - f(jj, :);
vp_pairs = (sum(response_diff < -erro, 2) + ...
    0.5 * sum(abs(response_diff) < erro, 2)) / size(f, 2);

v_r = sum((x - x_star).^2, 2);
distance_diff = v_r(ii) - v_r(jj);
vr_pairs = double(distance_diff < -erro) + ...
    0.5 * double(abs(distance_diff) < erro);

% Reconstruct the upper-triangular matrices before summing so MATLAB uses
% the same column-major accumulation order as the reference implementation.
vp = zeros(n, n);
vr = zeros(n, n);
linear_idx = ii + (jj - 1) * n;
vp(linear_idx) = vp_pairs;
vr(linear_idx) = vr_pairs;
sum_abs = sum(abs(vp - vr), 'all');
IL_ns = sum_abs / (n * (n + 1) / 2);
end
