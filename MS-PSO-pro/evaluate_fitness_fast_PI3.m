function fitness = evaluate_fitness_fast_PI3(particle, model, num_samples, dim, model_num, occupied_regions, roi_cache)
% evaluate_fitness_fast_PI 的 _PI3 变体(box3, 2026-08-26): 在 _PI 基础上加【目标启停】。
%
%   启停(box3-A): model.pi3_active = false(采样器判 α_eff < τ_drop, 先验整体失准)
%   时, 直接退回 2+has_class 个目标(IL, G, (U)), 当步完全不评估先验 —— 先验
%   连参赛资格都没有。active(默认: 只要 pi_phys_fn 存在)时与 _PI 逐字相同:
%
%   Phi_eff(R) = ( mean_{x∈R} pi_phys(x) ) ^ alpha_n
%
%   box3 采样器挂的 pi_phys_fn 已是退火场 π_eff = fused·anneal(box3-C),
%   本文件不感知退火细节(句柄透明传入)。
%
%   依赖 model 字段：
%     model.pi_phys_fn  先验句柄(active 时必需，否则报错)
%     model.pi_alpha    alpha_n（可选，默认 1）
%     model.pi3_active  目标启停（可选，默认 true）

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
    % --- [box3-A] 目标启停: inactive 时退回 2+has_class 目标 ---
    pi_on = isfield(model, 'pi_phys_fn') && ~isempty(model.pi_phys_fn);
    if pi_on && isfield(model, 'pi3_active')
        pi_on = logical(model.pi3_active);
    end
    num_fitness = 2 + has_class + pi_on;   % IL, G, (U), (Phi if active)
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

    % --- 物理目标 Phi_eff（仅 active 时追加，最大化）---
    if pi_on
        alpha_n = 1; if isfield(model,'pi_alpha') && ~isempty(model.pi_alpha), alpha_n = model.pi_alpha; end
        phi_R = mean(model.pi_phys_fn(sampled_points));   % 区域 R 上的平均先验强度(已含退火)
        objs(end+1) = phi_R ^ alpha_n;                     % 共享 alpha_n 注入
    end

    fitness = objs;
end
