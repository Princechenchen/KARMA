function [x_star, model] = HHs_Mixed(model)
% HHs_Mixed  Hollow-Hypersphere区域采样（混合响应改造版，C11）
%
% 文献来源：
%   Thedy ECH, Liao KW. Adaptive Kriging Adopting PSO with Hollow-Hypersphere
%   space in structural reliability assessment. Probabilistic Engineering
%   Mechanics, 2023, 74: 103513.
%   开源代码: https://github.com/johnthedy/KrigingPSO
%
% 本实现忠实保留了开源代码的核心机制，仅做混合响应适配：
%   保留的HHs核心机制（与开源代码一致）：
%     1. 球壳约束 bound()：径向投影（R<beta2→beta2，R>beta1→beta1）
%     2. CP线搜索 linesearch()：径向步进找决策边界交叉点
%     3. REIF学习函数 fobj()：mu(1-2Phi(mu/sig))+sig(w-sqrt(2/pi)exp(-mu^2/2sig^2)) × 概率密度
%     4. PSO连续优化 PSO_search()：惯性权重+c1+c2的完整PSO
%     5. beta1/beta2更新：beta1=CP+3_norm, beta2=0.95*CP
%     6. mode切换：REIF模式(mode=2) + 失效区深挖(mode=1)
%
%   混合响应改造点：
%     G(x) → 分类响应潜变量fmu（连续，fmu=0是决策边界）
%     标准正态空间‖U‖ → 原始[0,1]^d空间的归一化距离
%     mu_G, sigma_G → fmu（潜变量预测均值）, sigma_class（分类预测标准差）
%     标准正态概率密度 prod(phi(u_i)) → 均匀分布密度（恒定1，或用户自定义）
%     DACE Kriging → MMRGP predict_resp

    ns = size(model.candidate, 1);
    dim = model.dim;
    yd = size(model.init_value, 2);

    %% ---- 参数设置（与开源代码MAIN.m一致）----
    if isfield(model, 'hhs_beta1_init')
        beta1 = model.hhs_beta1_init;
    else
        beta1 = 0.7;  % 外半径（归一化到[0,1]^d空间）
    end
    if isfield(model, 'hhs_beta2_init')
        beta2 = model.hhs_beta2_init;
    else
        beta2 = 0.15;  % 内半径
    end

    % PSO参数（与开源PSO1.m一致）
    if isfield(model, 'hhs_pso_iters')
        pso_iters = model.hhs_pso_iters;
    else
        pso_iters = max(10, dim * 5);  % 开源：ite = nRV*5
    end
    if isfield(model, 'hhs_pso_particles')
        pso_particles = model.hhs_pso_particles;
    else
        pso_particles = max(15, dim * 15);  % 开源：ecosize = nRV*15
    end
    pso_weight = 0.95;  % 开源：weight = 0.95
    pso_c1 = 0.2;       % 开源：c1 = 0.2
    pso_c2 = 0.2;       % 开源：c2 = 0.2

    % CP线搜索参数
    delta_r = 0.02;     % 径向步长（开源：0.1，但[0,1]空间更小所以缩小）
    dmin = 0.01;        % 最小距离约束（开源：dmin = 0.01）

    % mode切换概率（开源：modelimit）
    if isfield(model, 'hhs_modelimit')
        modelimit = model.hhs_modelimit;
    else
        modelimit = 0.3;  % 开源默认值
    end

    % 空间中心和对角线
    center = 0.5 * ones(1, dim);
    diag_len = 0.5 * sqrt(dim);  % [0,1]^d空间中心到角点的距离

    %% ---- 1. MMRGP预测候选点 ----
    [y_pred, s_pred, model.MMRGP] = predict_resp(model.MMRGP, model.candidate);

    %% ---- 2. 提取分类响应潜变量fmu ----
    class_cols = find(model.ClassPos == 2);
    n_class = length(class_cols);

    if n_class == 0
        warning('HHs_Mixed: no classification response; using fallback.');
        % 无分类响应时用第一个连续响应的预测均值作为伪fmu
        fmu_cand = y_pred(:, 1);
        sigma_class_cand = sqrt(max(s_pred(1:yd:end, 1), 0));
    else
        if isfield(model.MMRGP, 'fmu') && ~isempty(model.MMRGP.fmu)
            fmu_cand = model.MMRGP.fmu(:, 1);  % 取第一个分类响应的fmu
            % 分类响应的预测标准差：从s_pred提取
            sigma_class_cand = sqrt(max(s_pred(class_cols(1):yd:end, class_cols(1)), 0));
        else
            fmu_cand = y_pred(:, class_cols(1)) * 2 - 1;
            sigma_class_cand = sqrt(max(s_pred(class_cols(1):yd:end, class_cols(1)), 0));
        end
    end

    %% ---- 3. CP线搜索（linesearch改造版）----
    train_x = model.now_x;
    n_train = size(train_x, 1);

    % 预测训练样本处的fmu
    [~, ~, tmp_model] = predict_resp(model.MMRGP, train_x);
    if n_class > 0 && isfield(tmp_model, 'fmu') && ~isempty(tmp_model.fmu)
        fmu_train = tmp_model.fmu(:, 1);
    else
        fmu_train = y_pred(1:n_train, 1);  % 回退
    end
    % 取训练样本的真实分类标签
    train_y_class = model.now_value(:, class_cols(1));
    % 确定失效类标签（用户可配置，默认0）
    if isfield(model, 'hhs_failure_class')
        failure_class = model.hhs_failure_class;
    else
        failure_class = 0;
    end
    if ~ismember(failure_class, unique(train_y_class(:))')
        error('HHs_Mixed: hhs_failure_class must appear in training labels.');
    end

    boundary_radii = [];
    for j = 1:n_train
        direction = train_x(j, :) - center;
        dist_j = norm(direction) / diag_len;
        if dist_j < 1e-8
            continue;
        end
        direction_unit = direction / norm(direction);
        fmu_j = fmu_train(j);

        % 用真实标签决定搜索方向：失效类向内（找安全），安全类向外（找失效）
        if train_y_class(j) == failure_class
            search_dir = -1;
        else
            search_dir = +1;
        end
        r_current = dist_j;
        for step = 1:100
            r_current = r_current + search_dir * delta_r;
            if r_current <= delta_r || r_current >= 1.0
                break;
            end
            search_point = center + direction_unit * (r_current * diag_len);
            search_point = min(max(search_point, 0), 1);
            [~, ~, sp_model] = predict_resp(model.MMRGP, search_point);
            if isfield(sp_model, 'fmu') && ~isempty(sp_model.fmu)
                fmu_sp = sp_model.fmu(1, 1);
            else
                break;
            end
            if sign(fmu_sp) ~= sign(fmu_j)
                boundary_radii = [boundary_radii; r_current];
                break;
            end
        end
    end

    %% ---- 4. 更新beta1/beta2（与开源linesearch2.m一致）----
    if ~isempty(boundary_radii)
        CP = min(boundary_radii);
        beta1 = min(CP + 0.3, 1.0);   % 开源：beta1 = CP+3（标准正态空间），这里缩小
        beta2 = max(CP * 0.95, 0.02); % 开源：beta2 = CP*0.95
    else
        if isfield(model, 'hhs_CP')
            CP = model.hhs_CP;
        else
            CP = 0.3;
        end
        beta1 = min(CP + 0.3, 1.0);
        beta2 = max(CP * 0.95, 0.02);  % 恢复：else分支也更新beta2
    end
    model.hhs_CP = CP;

    %% ---- 5. PSO搜索 + REIF学习函数（与开源fobjPSO.m + PSO1.m一致）----
    % 决定mode
    n_failure = sum(train_y_class == failure_class);
    if rand() < modelimit || n_failure < 1
        mode = 2;  % REIF模式
    else
        mode = 1;  % 失效区深挖模式
    end

    % 初始化PSO粒子在球壳内均匀分布（与开源KrigingPSO.m一致）
    pso_positions = initialize_shell_particles(pso_particles, dim, beta1, beta2, center, diag_len);
    pso_pbest = pso_positions;
    pso_velocities = zeros(pso_particles, dim);
    pso_pbest_fitness = inf(pso_particles, 1);
    pso_gbest = pso_positions(1, :);
    pso_gbest_fitness = inf;

    for iter = 1:pso_iters
        % 评估每个粒子的适应度（fobj改造版）
        for p = 1:pso_particles
            fit = fobj_hhs(pso_positions(p, :), fmu_cand, sigma_class_cand, ...
                            model.candidate, mode, dmin, CP, center, diag_len, model.now_x);
            if fit < pso_pbest_fitness(p)
                pso_pbest_fitness(p) = fit;
                pso_pbest(p, :) = pso_positions(p, :);
            end
            if fit < pso_gbest_fitness
                pso_gbest_fitness = fit;
                pso_gbest = pso_positions(p, :);
            end
        end

        % 更新速度和位置（与开源PSO1.m一致）
        r1 = rand(pso_particles, dim);
        r2 = rand(pso_particles, dim);
        pso_velocities = pso_weight * pso_velocities + ...
                         pso_c1 * r1 .* (pso_pbest - pso_positions) + ...
                         pso_c2 * r2 .* (repmat(pso_gbest, pso_particles, 1) - pso_positions);
        pso_positions = pso_positions + pso_velocities;

        % 球壳约束（与开源bound.m一致：径向投影）
        pso_positions = bound_positions(pso_positions, beta1, beta2, center, diag_len);

        % 10%粒子随机重采到球壳内（与开源PSO1.m一致）
        for p = 1:pso_particles
            if rand() < 0.1
                pso_positions(p, :) = random_shell_point(dim, beta1, beta2, center, diag_len);
            end
        end
        % 保留gbest（开源：x(1,:)=x_gbest）
        pso_positions(1, :) = pso_gbest;
    end

    %% ---- 6. 将PSO最优点映射到最近候选点 ----
    pso_gbest = min(max(pso_gbest, 0), 1);  % 裁剪到[0,1]
    [~, best_idx] = min(pdist2(model.candidate, pso_gbest));
    x_star = model.candidate(best_idx, :);

    %% ---- 7. 记录诊断信息（恢复）----
    model.HHs = struct('beta1', beta1, 'beta2', beta2, 'CP', CP, ...
                       'mode', mode, 'n_boundary', length(boundary_radii), ...
                       'gbest', pso_gbest, 'gbest_fitness', pso_gbest_fitness);

    if isfield(model, 'hhs_beta1_history') && ~isempty(model.hhs_beta1_history)
        model.hhs_beta1_history(end+1) = beta1;
        model.hhs_beta2_history(end+1) = beta2;
        model.hhs_CP_history(end+1) = CP;
    else
        model.hhs_beta1_history = beta1;
        model.hhs_beta2_history = beta2;
        model.hhs_CP_history = CP;
    end

    %% ---- 8. 记录采样标记（恢复）----
    model.M_ind(size(model.now_value, 1) - size(model.init_value, 1) + 1) = 0;
end

%% ==================== 辅助函数 ====================

function particles = initialize_shell_particles(n, dim, beta1, beta2, center, diag_len)
% 在球壳内均匀初始化粒子（与开源KrigingPSO.m的初始化一致）
    particles = zeros(n, dim);
    for i = 1:n
        particles(i, :) = random_shell_point(dim, beta1, beta2, center, diag_len);
    end
end

function pt = random_shell_point(dim, beta1, beta2, center, diag_len)
% 在球壳beta2<=r<=beta1内随机生成一个点
    dummy = unifrnd(-1, 1, 1, dim);
    r = unifrnd(beta2, beta1);
    pt = center + dummy .* (r * diag_len / (norm(dummy) + 1e-12));
    pt = min(max(pt, 0), 1);
end

function x = bound_positions(x, beta1, beta2, center, diag_len)
% 球壳径向投影（与开源bound.m一致）
    for i = 1:size(x, 1)
        diff = x(i, :) - center;
        R = norm(diff) / diag_len;
        if R < beta2 && R > 1e-12
            x(i, :) = center + diff .* (beta2 / R);
        elseif R > beta1
            x(i, :) = center + diff .* (beta1 / R);
        end
        x(i, :) = min(max(x(i, :), 0), 1);
    end
end

function y = fobj_hhs(x_pos, fmu_cand, sigma_cand, candidate, mode, dmin, CP, center, diag_len, xdoe)
% REIF学习函数改造版（与开源fobjPSO.m一致）
% 输入x_pos是PSO粒子的连续位置，需要在候选点中插值获取fmu和sigma

    % 找最近的候选点获取预测值
    [~, idx] = min(pdist2(candidate, x_pos));
    mu = fmu_cand(idx);
    sig = sigma_cand(idx);

    if sig < 1e-12
        sig = 1e-6;
    end

    w = 2;  % 开源参数

    % REIF公式（开源fobjPSO.m）：mu(1-2Phi(mu/sig))+sig(w-sqrt(2/pi)exp(-mu^2/2sig^2))
    reif = mu * (1 - 2 * normcdf(mu / sig)) + sig * (w - sqrt(2/pi) * exp(-0.5 * (mu/sig)^2));

    % 概率密度加权（开源：fp = prod(normpdf(x,0,1))）
    % 改造：均匀分布下fp=1（恒定）
    fp = 1;  % 均匀输入空间

    y = 1e9 - reif * fp;  % 开源：y = 1e9 - y.*fp（最小化→最大化REIF）

    % mode=1：失效区深挖（开源fobjPSO.m的mode=1逻辑）
    if mu <= 0 && mode == 1
        dist_to_CP = abs(norm(x_pos - center) / diag_len - CP);
        N_near = sum(pdist2(xdoe, x_pos) < dmin * 3);
        w1 = 10; w2 = 1; w3 = 10;
        y = -1 / (w1 * abs(mu) * dist_to_CP + w2 * dist_to_CP + w3 * N_near);
    end

    % 距离约束（开源：min||x-xdoe||<dmin → Inf）
    if min(pdist2(xdoe, x_pos)) < dmin
        y = Inf;
    end
end
