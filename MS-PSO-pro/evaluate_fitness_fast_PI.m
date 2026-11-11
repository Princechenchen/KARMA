function fitness = evaluate_fitness_fast_PI(particle, model, num_samples, dim, model_num, occupied_regions, roi_cache)
% evaluate_fitness_fast 的 _PI 变体：在原 IL/G/(U) 基础上追加【物理目标 Phi】。
%
%   Phi_eff(R) = ( mean_{x∈R} pi_phys(x) ) ^ alpha_n
%
%   —— 区域平均先验强度作为 MOPSO 第 4 目标（最大化），并通过幂次 alpha_n
%   注入"共享可信度"：alpha_n→0 时 Phi_eff→1（物理目标失去区分力，L1 与 L2
%   一起放松）；alpha_n=1 时 Phi_eff=Phi。alpha_n 与 L2 采样函数共用同一个值。
%
%   依赖 model 字段：
%     model.pi_phys_fn  先验句柄（必需，否则报错）
%     model.pi_alpha    alpha_n（可选，默认 1）

    % --- 区域解码（与原版一致）---
    region_bounds = zeros(dim, 2);
    new_region = zeros(1, 2 * dim);
    for d = 1:dim
        feature_start = particle(2 * d - 1);
        feature_end = feature_start + particle(2 * d);
        region_bounds(d, :) = [feature_start, feature_end];
        new_region(2 * d - 1:2 * d) = [feature_start, feature_end];
    end

    has_class = isfield(model, 'ClassPos') && any(model.ClassPos == 2);
    num_fitness = 3 + has_class;   % IL, G, (U), Phi
    if is_overlapping(new_region, occupied_regions)
        fitness = -1e6 * ones(1, num_fitness);
        return;
    end

    % --- 复用原 LHS 采样点计算 IL/G/(U) ---
    sampled_points = latin_hypercube_sampling(region_bounds, num_samples);
    [sampledY, ~] = predict_resp_roi_fast(model.MMRGP, sampled_points, roi_cache);
    information_surface_measure = calculate_entropy(sampled_points, sampledY, 0.05);
    gradient_metric = compute_gradient_metric(sampled_points, sampledY);

    objs = [information_surface_measure, gradient_metric];
    if has_class
        class_cols = find(model.ClassPos == 2);
        uncertainty_sum = 0;
        for k = 1:numel(class_cols)
            p = (sampledY(:, class_cols(k)) + 1) / 2;
            p = min(max(p, 1e-12), 1 - 1e-12);
            entropy_value = -(p .* log(p) + (1 - p) .* log(1 - p));
            uncertainty_sum = uncertainty_sum + mean(entropy_value) / log(2);
        end
        objs(end+1) = uncertainty_sum / numel(class_cols);
    end

    % --- 物理目标 Phi_eff（最后一个目标，最大化）---
    if ~isfield(model,'pi_phys_fn') || isempty(model.pi_phys_fn)
        error('evaluate_fitness_fast_PI: model.pi_phys_fn 缺失（_PI 模式必须提供先验）');
    end
    alpha_n = 1; if isfield(model,'pi_alpha') && ~isempty(model.pi_alpha), alpha_n = model.pi_alpha; end
    phi_R = mean(model.pi_phys_fn(sampled_points));   % 区域 R 上的平均先验强度
    objs(end+1) = phi_R ^ alpha_n;                     % 共享 alpha_n 注入

    fitness = objs;
end
