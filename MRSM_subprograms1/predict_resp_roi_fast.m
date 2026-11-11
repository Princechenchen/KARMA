function [y, fc] = predict_resp_roi_fast(model, x, cache)
% Predict only the response quantities used by ROI fitness evaluation.
z_cross = define_covmatrix0(model, x, model.X);
f0 = trend_fun(model, x);
y_normalized = f0 * model.beta + z_cross * cache.alpha2;
y_normalized = reshape(y_normalized, model.m, size(x, 1))';

fc = [];
if ~isempty(cache.class_cols)
    % predict_resp uses the first prediction block's variance for all points.
    u = f0 - z_cross * cache.alpha3;
    % Keep the original multi-RHS solve shape so BLAS rounding is identical.
    alpha5 = cache.alpha4chol \ (cache.alpha4chol' \ u');
    alpha6 = cache.zchol \ (cache.zchol' \ z_cross');
    class_row = cache.class_cols(1);
    z0 = model.hyper.A * model.hyper.A';
    fs2 = z0(class_row, class_row) ...
        - z_cross(class_row, :) * alpha6(:, class_row) ...
        + u(class_row, :) * alpha5(:, class_row);
    fmu = y_normalized(:, cache.class_cols);
    fc = 2 * exp(logphi(fmu ./ sqrt(1 + fs2))) - 1;
end

settings = cache.settings;
y = (y_normalized' - settings.ymin) ./ settings.gain + settings.xoffset;
y = y';
if ~isempty(cache.class_cols)
    y(:, cache.class_cols) = fc;
end
end
