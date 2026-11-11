function [y, fc] = predict_resp_roi_groups_fast(model, x, group_starts, cache)
% Predict one concatenated generation while preserving per-particle fs2 values.
z_cross = define_covmatrix0(model, x, model.X);
f0 = trend_fun(model, x);
y_normalized = f0 * model.beta + z_cross * cache.alpha2;
y_normalized = reshape(y_normalized, model.m, size(x, 1))';

u = f0 - z_cross * cache.alpha3;
alpha5 = cache.alpha4chol \ (cache.alpha4chol' \ u');
alpha6 = cache.zchol \ (cache.zchol' \ z_cross');
class_col = cache.class_cols(1);
z0 = model.hyper.A * model.hyper.A';
fc = zeros(size(x, 1), 1);

group_ends = [group_starts(2:end) - 1; size(x, 1)];
for g = 1:numel(group_starts)
    point_rows = group_starts(g):group_ends(g);
    covariance_row = (group_starts(g) - 1) * model.m + class_col;
    fs2 = z0(class_col, class_col) ...
        - z_cross(covariance_row, :) * alpha6(:, covariance_row) ...
        + u(covariance_row, :) * alpha5(:, covariance_row);
    fmu = y_normalized(point_rows, class_col);
    fc(point_rows) = 2 * exp(logphi(fmu ./ sqrt(1 + fs2))) - 1;
end

settings = cache.settings;
y = (y_normalized' - settings.ymin) ./ settings.gain + settings.xoffset;
y = y';
y(:, class_col) = fc;
end
