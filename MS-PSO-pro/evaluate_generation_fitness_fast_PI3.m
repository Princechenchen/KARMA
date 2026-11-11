function fitnesses = evaluate_generation_fitness_fast_PI3(positions, model, ...
    num_samples, dim, model_num, occupied_regions, roi_cache, unit_template, vectorized_metrics)
% evaluate_generation_fitness_fast_PI 的 _PI3 变体(box3, 2026-08-26):
% 在 _PI 基础上加【目标启停 box3-A】—— model.pi3_active=false 时退回
% 2+has_class 目标(IL,G,(U)), 当步不评估任何先验; active 时与 _PI 逐字相同:
%
%   Phi_eff(R) = ( mean_{x∈R} pi_phys(x) ) ^ alpha_n   （见 evaluate_fitness_fast_PI3）
%   先验对所有 feasible 粒子的采样点一次性批量求值（pi_fn(all_samples)），按组取 mean。
%   box3 采样器挂的 pi_phys_fn 已是退火场 π_eff = fused·anneal(box3-C)。
%
%   依赖 model 字段：model.pi_phys_fn（active 时必需）、model.pi_alpha（可选，默认 1）、
%   model.pi3_active（可选，默认 true）。
    if nargin < 9
        vectorized_metrics = false;
    end
    if nargin < 8
        unit_template = [];
    end
    num_particles = size(positions, 1);
    has_class = isfield(model, 'ClassPos') && any(model.ClassPos == 2);
    % --- [box3-A] 目标启停 ---
    pi_on = isfield(model, 'pi_phys_fn') && ~isempty(model.pi_phys_fn);
    if pi_on && isfield(model, 'pi3_active')
        pi_on = logical(model.pi3_active);
    end
    num_fitness = 2 + has_class + pi_on;   % IL, G, (U), (Phi if active)
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

    % --- PI 先验一次性批量求值(仅 active) ---
    all_phi = [];
    if pi_on
        if ~isfield(model,'pi_phys_fn') || isempty(model.pi_phys_fn)
            error('evaluate_generation_fitness_fast_PI3: model.pi_phys_fn 缺失（_PI3 active 模式必须提供先验）');
        end
        alpha_n = 1; if isfield(model,'pi_alpha') && ~isempty(model.pi_alpha), alpha_n = model.pi_alpha; end
        all_phi = model.pi_phys_fn(all_samples);   % [sum_samples × 1](已含退火)
    end

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

        objs = [information_surface_measure, gradient_metric];
        if has_class
            uncertainty_sum = 0;
            for k = 1:numel(class_cols)
                p = (sampledY(:, class_cols(k)) + 1) / 2;
                p = min(max(p, 1e-12), 1 - 1e-12);
                entropy_value = -(p .* log(p) + (1 - p) .* log(1 - p));
                uncertainty_sum = uncertainty_sum + mean(entropy_value) / log(2);
            end
            objs(end+1) = uncertainty_sum / numel(class_cols);
        end
        if pi_on
            phi_R = mean(all_phi(rows));            % 该组（区域 R）平均先验强度
            objs(end+1) = phi_R ^ alpha_n;          % 共享 alpha_n 注入
        end
        fitnesses(i, :) = objs;
    end
end
