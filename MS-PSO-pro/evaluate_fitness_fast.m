function fitness = evaluate_fitness_fast(particle, model, num_samples, dim, model_num, occupied_regions, roi_cache)
% Evaluate the original ROI objectives using the cached prediction path.
region_bounds = zeros(dim, 2);
new_region = zeros(1, 2 * dim);
for d = 1:dim
    feature_start = particle(2 * d - 1);
    feature_end = feature_start + particle(2 * d);
    region_bounds(d, :) = [feature_start, feature_end];
    new_region(2 * d - 1:2 * d) = [feature_start, feature_end];
end

has_class = isfield(model, 'ClassPos') && any(model.ClassPos == 2);
num_fitness = 2 + has_class;
if is_overlapping(new_region, occupied_regions)
    fitness = -1e6 * ones(1, num_fitness);
    return;
end

sampled_points = latin_hypercube_sampling(region_bounds, num_samples);
[sampledY, ~] = predict_resp_roi_fast(model.MMRGP, sampled_points, roi_cache);
information_surface_measure = calculate_entropy(sampled_points, sampledY, 0.05);
gradient_metric = compute_gradient_metric(sampled_points, sampledY);

class_uncertainty = [];
if has_class
    class_cols = find(model.ClassPos == 2);
    uncertainty_sum = 0;
    for k = 1:numel(class_cols)
        p = (sampledY(:, class_cols(k)) + 1) / 2;
        p = min(max(p, 1e-12), 1 - 1e-12);
        entropy_value = -(p .* log(p) + (1 - p) .* log(1 - p));
        uncertainty_sum = uncertainty_sum + mean(entropy_value) / log(2);
    end
    class_uncertainty = uncertainty_sum / numel(class_cols);
end

fitness = [information_surface_measure, gradient_metric, class_uncertainty];
end
