function fitnesses = evaluate_generation_fitness_fast(positions, model, ...
    num_samples, dim, model_num, occupied_regions, roi_cache, unit_template, vectorized_metrics)
% Evaluate a generation with one concatenated cached MMRGP prediction.
if nargin < 9
    vectorized_metrics = false;
end
if nargin < 8
    unit_template = [];
end
num_particles = size(positions, 1);
has_class = isfield(model, 'ClassPos') && any(model.ClassPos == 2);
num_fitness = 2 + has_class;
fitnesses = -1e6 * ones(num_particles, num_fitness);
sample_cells = cell(num_particles, 1);
new_regions = zeros(num_particles, 2 * dim);

for i = 1:num_particles
    region_bounds = zeros(dim, 2);
    new_region = zeros(1, 2 * dim);
    for d = 1:dim
        feature_start = positions(i, 2 * d - 1);
        feature_end = feature_start + positions(i, 2 * d);
        region_bounds(d, :) = [feature_start, feature_end];
        new_region(2 * d - 1:2 * d) = [feature_start, feature_end];
    end
    new_regions(i, :) = new_region;
end

if vectorized_metrics
    feasible = ~is_overlapping_batch(new_regions, occupied_regions);
else
    feasible = false(num_particles, 1);
    for i = 1:num_particles
        feasible(i) = ~is_overlapping(new_regions(i, :), occupied_regions);
    end
end

for i = find(feasible)'
    region_bounds = [new_regions(i, 1:2:end)', new_regions(i, 2:2:end)'];
    if isempty(unit_template)
        sample_cells{i} = latin_hypercube_sampling(region_bounds, num_samples);
    else
        sample_cells{i} = latin_hypercube_sampling(region_bounds, num_samples, unit_template);
    end
end

feasible_idx = find(feasible);
if isempty(feasible_idx)
    return;
end
all_samples = vertcat(sample_cells{feasible_idx});
group_starts = (0:numel(feasible_idx)-1)' * num_samples + 1;
[all_predictions, ~] = predict_resp_roi_groups_fast( ...
    model.MMRGP, all_samples, group_starts, roi_cache);

class_cols = find(model.ClassPos == 2);
for g = 1:numel(feasible_idx)
    i = feasible_idx(g);
    rows = group_starts(g):group_starts(g) + num_samples - 1;
    sampled_points = sample_cells{i};
    sampledY = all_predictions(rows, :);
    if vectorized_metrics
        information_surface_measure = calculate_entropy_fast(sampled_points, sampledY, 0.05);
    else
        information_surface_measure = calculate_entropy(sampled_points, sampledY, 0.05);
    end
    gradient_metric = compute_gradient_metric(sampled_points, sampledY);
    class_uncertainty = [];
    if has_class
        uncertainty_sum = 0;
        for k = 1:numel(class_cols)
            p = (sampledY(:, class_cols(k)) + 1) / 2;
            p = min(max(p, 1e-12), 1 - 1e-12);
            entropy_value = -(p .* log(p) + (1 - p) .* log(1 - p));
            uncertainty_sum = uncertainty_sum + mean(entropy_value) / log(2);
        end
        class_uncertainty = uncertainty_sum / numel(class_cols);
    end
    fitnesses(i, :) = [information_surface_measure, gradient_metric, class_uncertainty];
end
end
