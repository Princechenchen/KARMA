function [occupied_table, diagnostics] = mopso_adaptive_elitism_crowding_snap(dim, bounds, window_size_bounds, ...
    model, num_particles, iterations1, iterations2, num_samples, elitism_size, model_num, roi_cache, fast_options, snap_gens)
% mopso_adaptive_elitism_crowding_snap —— 挖矿插桩 fork(2026-09-03, 原文件零改动)
% 与原版逐位一致, 仅两处差异: ① 前 snap_gens 代结束时记录非支配存档快照;
% ② 跑满 snap_gens 代后提前退出(只为挖早期代际 ROI 窗口, 不用于正式采样)。
if nargin < 13 || isempty(snap_gens), snap_gens = 5; end

    use_fast_roi_prediction = nargin >= 11 && ~isempty(roi_cache);
    batch_prediction = false;
    batch_archive = false;
    reuse_sampling_template = false;
    vectorized_roi_metrics = false;
    if nargin >= 12 && ~isempty(fast_options)
        if isfield(fast_options, 'batch_prediction')
            batch_prediction = logical(fast_options.batch_prediction);
        end
        if isfield(fast_options, 'batch_archive')
            batch_archive = logical(fast_options.batch_archive);
        end
        if isfield(fast_options, 'reuse_sampling_template')
            reuse_sampling_template = logical(fast_options.reuse_sampling_template);
        end
        if isfield(fast_options, 'vectorized_roi_metrics')
            vectorized_roi_metrics = logical(fast_options.vectorized_roi_metrics);
        end
    end
    if use_fast_roi_prediction
        validate_roi_predict_cache(model.MMRGP, roi_cache);
    end
    
    % 初始化参数记录结构
    parameter_metrics = struct('inertia_weight', [], 'cognitive_factor', [], 'social_factor', []);
    
    % 新增：初始化占用区域和适应度存储
    occupied_regions = [];
    occupied_fitnesses = cell(0);
    if isfield(model, 'ClassPos') && any(model.ClassPos == 2)
        num_fitness = 3;  % IL, G, class uncertainty (NME)
    else
        num_fitness = 2;  % IL, G
    end

    % 由 iterations2 反推 RoI_num（兼容原接口）
    % 注意：iterations2 现同时作为自适应重启的耐心值（patience）
    original_iterations1 = iterations1;
    RoI_num = max(1, round(original_iterations1 / iterations2));
    max_archive_size = num_particles;%max(50, 3 * RoI_num); % 最大存档数量（三目标 + 跨周期累积需要更大容量）

    % 速度限制计算
    v_max = [];
    for d = 1:dim
        v_max = [v_max, (bounds(d,2)-bounds(d,1))*0.2];
        v_max = [v_max, (window_size_bounds(d,2)-window_size_bounds(d,1))*0.2];
    end

    % 边界向量准备（原固定边界方式已注释，现改用 size-first 初始化）
    % l_bounds = []; u_bounds = [];
    % for d = 1:dim
    %     l_bounds = [l_bounds, bounds(d,1), window_size_bounds(d,1)];
    %     u_bounds = [u_bounds, bounds(d,2)-window_size_bounds(d,2), window_size_bounds(d,2)];
    % end

    % 初始化粒子群（带重叠检测，size-first：先确定窗口宽度，再确定起始位置）
    max_attempts = 10;  % 最大尝试次数
    valid_count = 0;
    total_generated = 0;
    particles = struct('position', {}, 'velocity', {}, 'personal_best_position', {}, 'personal_best_fitness', {});

    for attempt = 1:max_attempts
        % 每次生成双倍样本（可根据需要调整倍数）
        sampler = lhsdesign(num_particles*2, 2*dim);

        % 处理当前批次的样本
        for i = 1:size(sampler,1)
            % 位置生成（size-first：先随机窗口大小，再根据大小决定起始位置范围）
            pos = zeros(1, 2*dim);
            for d = 1:dim
                % 先生成窗口大小
                size_val = window_size_bounds(d,1) + ...
                    (window_size_bounds(d,2) - window_size_bounds(d,1)) * sampler(i, 2*d);
                % 起始位置范围取决于当前窗口大小：[lower, upper - size]
                start_max = bounds(d,2) - size_val;
                pos(2*d-1) = bounds(d,1) + (start_max - bounds(d,1)) * sampler(i, 2*d-1);
                pos(2*d) = size_val;
            end

            % 构造区域表示
            region = zeros(1, 2*dim);
            for d = 1:dim
                region(2*d-1) = pos(2*d-1);
                region(2*d) = pos(2*d-1) + pos(2*d);
            end

            % 重叠检测
            if ~is_overlapping(region, occupied_regions)
                valid_count = valid_count + 1;

                % 存储粒子（完整初始化所有字段）
                particles(valid_count).position = pos;
                particles(valid_count).velocity = -v_max + 2*v_max.*rand(1,2*dim);
                particles(valid_count).personal_best_position = pos;
                particles(valid_count).personal_best_fitness = -inf(1,num_fitness);

                % 达到需求数量则退出
                if valid_count >= num_particles
                    break;
                end
            end
        end

        % 检查是否完成初始化
        if valid_count >= num_particles
            break;  % 成功生成足够粒子，退出尝试循环
        else
            fprintf('尝试 %d: 成功生成 %d/%d 粒子\n', attempt, valid_count, num_particles);
        end
    end

    % 最终验证
    if valid_count == 0
        error('初始化失败：经过%d次尝试，无法找到非重叠粒子', max_attempts);
    elseif valid_count < num_particles
        warning('部分初始化：仅成功生成 %d/%d 粒子', valid_count, num_particles);
        num_particles = valid_count;  % 更新实际粒子数
        particles = particles(1:num_particles);  % 截断数组
    end

    % 动态参数设置（自适应认知/社会系数 + 线性递减惯性权重）
    % 原 tan 调度参数已注释，保留供参考
    % w0 = 0.72; wK = 1.17;
    % c1_0 = 1.54; c1_K = 2.07;
    % c2_0 = 1.31; c2_K = 2.86;
    % delta = 0.1;

    % 线性递减惯性权重参数
    w_max = 0.9;
    w_min = 0.4;
    % 自适应认知/社会系数参数
    c1_max = 2.5;  c1_min = 0.5;
    c2_max = 2.5;  c2_min = 0.5;

    % 初始化存档和相关指标
    archive = [];
    archive_fitnesses = cell(0);
    diversity_metrics = struct('variance', [], 'coverage', []);
    coverage_radius_metrics = [];
    uniform_distribution_metrics = [];
    iterations1 = iterations1 + 1;

    % 初始化全局最优解
    global_best = particles(1).position;

    % 将粒子数据转换为矩阵形式以便向量化操作
    positions = vertcat(particles.position);
    velocities = vertcat(particles.velocity);
    personal_best_positions = vertcat(particles.personal_best_position);
    personal_best_fitnesses = vertcat(particles.personal_best_fitness);

    % 预分配内存 - 使用实际粒子数量而不是初始值
    parameter_metrics.inertia_weight = zeros(iterations1, 1);
    parameter_metrics.cognitive_factor = zeros(iterations1, 1);
    parameter_metrics.social_factor = zeros(iterations1, 1);
    fitnesses = zeros(num_particles, num_fitness);

    % 记录实际完成的迭代次数
    actual_iterations = 0;

    % 自适应重启参数（基于 Hypervolume 的停滞检测）
    % 耐心值 patience：连续 patience 代 Hypervolume 无明显提升则重启
    patience = max(10, round(iterations2 / 5));
    hypervolume_improvement_threshold = 1e-3;  % 相对 Hypervolume 提升阈值
    min_archive_size_for_restart = 5;          % archive 至少达到该规模才允许重启
    stagnation_counter = 0;
    blank_restart_counter = 0;                 % 连续未找到新 ROI 的重启次数
    max_blank_restarts = 3;                    % 连续 3 次找不到新 ROI 就停止重启
    stop_restarting = false;                   % 是否停止重启
    last_hypervolume = 0;                      % 上一轮 Hypervolume
    ref_point = [];                            % Hypervolume 参考点（首次计算时固定）
    iter_since_restart = 0;                    % 当前 ROI 内已迭代次数
    roi_collected = 0;
    diagnostics = struct('archive_size', zeros(iterations1, 1), ...
        'archive_truncations', zeros(iterations1, 1), ...
        'hypervolume', nan(iterations1, 1), ...
        'stagnation_counter', zeros(iterations1, 1), ...
        'restart', false(iterations1, 1), ...
        'roi_collected', zeros(iterations1, 1));

    unit_template = [];
    if use_fast_roi_prediction && batch_prediction && reuse_sampling_template
        rng_state = rng;
        unit_template = lhsdesign(num_samples, dim, 'criterion', 'none');
        rng(rng_state);
    end

    % 主优化循环
    for t = 1:iterations1
        tic;
        actual_iterations = actual_iterations + 1;

        % 动态参数计算（自适应认知/社会系数 + 线性递减惯性权重）
        % 每次重启后按当前 ROI 的进度重新调度，避免后期重启直接陷入 exploitation
        ratio = min(iter_since_restart / max(iterations2, 1), 1);
        % 原 tan 调度已注释，保留供参考
        % w = (w0 - wK) * tan((1 - ratio) + delta) + wK;
        % c1 = (c1_0 - c1_K) * tan(ratio + delta) + c1_K;
        % c2 = (c2_0 - c2_K) * tan((1 - ratio) + delta) + c2_K;

        % 线性递减惯性权重
        w = w_max - (w_max - w_min) * ratio;
        % 自适应认知系数：前期大、后期小
        c1 = c1_max - (c1_max - c1_min) * ratio;
        % 自适应社会系数：前期小、后期大
        c2 = c2_min + (c2_max - c2_min) * ratio;

        % 记录参数
        parameter_metrics.inertia_weight(t) = w;
        parameter_metrics.cognitive_factor(t) = c1;
        parameter_metrics.social_factor(t) = c2;

        % 当前 ROI 内迭代计数递增
        iter_since_restart = iter_since_restart + 1;

        % 在parfor循环之前，显式创建所有需要的临时变量
        temp_model = model;
        % temp_kde_q_ = kde_q_;
        temp_num_samples = num_samples;
        temp_dim = dim;
        temp_model_num = model_num;
        % temp_all_extrema = all_extrema;
        temp_occupied_regions = occupied_regions; % 创建常量副本

        % 评估适应度（使用 for 循环以避免并行池内存开销）
        if use_fast_roi_prediction && batch_prediction
            fitnesses = evaluate_generation_fitness_fast(positions, temp_model, ...
                temp_num_samples, temp_dim, temp_model_num, temp_occupied_regions, ...
                roi_cache, unit_template, vectorized_roi_metrics);
        else
            for i = 1:num_particles
                if use_fast_roi_prediction
                    fitnesses(i,:) = evaluate_fitness_fast(positions(i,:), temp_model, ...
                        temp_num_samples, temp_dim, temp_model_num, temp_occupied_regions, roi_cache);
                else
                    fitnesses(i,:) = evaluate_fitness(positions(i,:), temp_model, ...
                        temp_num_samples, temp_dim, temp_model_num, temp_occupied_regions);
                end
            end
        end

        % 更新个体最优
        truncations_this_generation = 0;
        for i = 1:num_particles
            if dominates(fitnesses(i,:), personal_best_fitnesses(i,:))
                personal_best_positions(i,:) = positions(i,:);
                personal_best_fitnesses(i,:) = fitnesses(i,:);
            end

            % 更新存档
        end
        if use_fast_roi_prediction && batch_archive
            [archive, archive_fitnesses, did_truncate] = update_archive_batch( ...
                archive, archive_fitnesses, positions, fitnesses, max_archive_size);
            truncations_this_generation = double(did_truncate);
        else
            for i = 1:num_particles
                [archive, archive_fitnesses, did_truncate] = update_archive(archive, archive_fitnesses, ...
                    positions(i,:), fitnesses(i,:), max_archive_size);
                truncations_this_generation = truncations_this_generation + double(did_truncate);
            end
        end
        diagnostics.archive_size(t) = size(archive, 1);
        diagnostics.archive_truncations(t) = truncations_this_generation;

        % === [snap 插桩] 前 snap_gens 代记录非支配存档窗口 ===
        if t <= snap_gens
            diagnostics.snap{t} = struct('gen', t, ...
                'archive', archive, 'fitnesses', {archive_fitnesses});
        end
        if t >= snap_gens
            break;   % 提前退出: 只挖早期代际
        end

        % --- 自适应停滞检测（基于 Hypervolume）---
        if ~isempty(archive_fitnesses) && size(archive, 1) >= min_archive_size_for_restart
            front = cell2mat(archive_fitnesses);

            % 首次计算时确定参考点，之后固定使用
            % 注意：本算法为最大化问题，参考点应比所有目标值都小
            if isempty(ref_point)
                ref_point = min(front, [], 1) - 0.1 * (max(front, [], 1) - min(front, [], 1));
            end

            current_hypervolume = compute_hypervolume(front, ref_point, 20000, 'max');
            diagnostics.hypervolume(t) = current_hypervolume;

            if last_hypervolume == 0
                % 首次达到足够 archive 规模，初始化基准 Hypervolume
                last_hypervolume = current_hypervolume;
                stagnation_counter = 0;
            else
                relative_improvement = (current_hypervolume - last_hypervolume) / max(abs(last_hypervolume), 1e-12);
                if relative_improvement > hypervolume_improvement_threshold
                    % Hypervolume 仍在增长，不判为停滞
                    stagnation_counter = 0;
                    last_hypervolume = current_hypervolume;
                    blank_restart_counter = 0;
                    stop_restarting = false;
                else
                    % Hypervolume 连续未增长
                    stagnation_counter = stagnation_counter + 1;
                end
            end
        else
            % archive 太小，暂不判断停滞
            stagnation_counter = 0;
        end

        % 全局最优选择（每轮迭代执行）
        % 从 archive 中筛选不与 occupied_regions 重叠的可行解，
        % 然后在第一前沿解中使用二元锦标赛选择全局最优：
        % 30% 按拥挤度（多样性），70% 按三目标等权质量分（收敛性）
        diagnostics.stagnation_counter(t) = stagnation_counter;

        if ~isempty(archive_fitnesses)
            n_archive = size(archive, 1);
            archive_regions = zeros(n_archive, 2 * dim);
            archive_regions(:, 1:2:end) = archive(:, 1:2:end);
            archive_regions(:, 2:2:end) = archive(:, 1:2:end) + archive(:, 2:2:end);
            if vectorized_roi_metrics
                feasible_idx = ~is_overlapping_batch(archive_regions, occupied_regions);
            else
                feasible_idx = true(n_archive, 1);
                for i = 1:n_archive
                    feasible_idx(i) = ~is_overlapping(archive_regions(i,:), occupied_regions);
                end
            end

            if any(feasible_idx)
                feasible_fitnesses = archive_fitnesses(feasible_idx);
                fronts = non_dominated_sorting(feasible_fitnesses);
                first_front_local = fronts{1};
                first_front_all = find(feasible_idx);
                first_front = first_front_all(first_front_local);

                % 计算拥挤度
                crowding_distances = compute_crowding_distance(archive_fitnesses(first_front));

                % 计算三目标等权质量分（min-max 归一化）
                fit_matrix = cell2mat(archive_fitnesses(first_front));
                fit_min = min(fit_matrix, [], 1);
                fit_max = max(fit_matrix, [], 1);
                fit_range = fit_max - fit_min;
                fit_range(fit_range == 0) = 1;
                quality_scores = mean((fit_matrix - fit_min) ./ fit_range, 2);
%                 quality_scores = sum((fit_matrix - fit_min) ./ fit_range.*[0.3,0.3,0.4],2);
                % 二元锦标赛
                n_front = length(first_front);
                idx1 = randi(n_front);
                idx2 = randi(n_front);
                while idx2 == idx1 && n_front > 1
                    idx2 = randi(n_front);
                end

                if rand < 0.3
                    % 30%：拥挤度大者胜（多样性）
                    if crowding_distances(idx1) >= crowding_distances(idx2)
                        winner = idx1;
                    else
                        winner = idx2;
                    end
                else
                    % 70%：质量分高者胜（收敛性）
                    if quality_scores(idx1) >= quality_scores(idx2)
                        winner = idx1;
                    else
                        winner = idx2;
                    end
                end

                global_best = archive(first_front(winner), :);
            else
                % 没有可行解时，从整个 archive 中选质量分最高者
                fit_matrix = cell2mat(archive_fitnesses);
                fit_min = min(fit_matrix, [], 1);
                fit_max = max(fit_matrix, [], 1);
                fit_range = fit_max - fit_min;
                fit_range(fit_range == 0) = 1;
                quality_scores = mean((fit_matrix - fit_min) ./ fit_range, 2);
                [~, best_idx] = max(quality_scores);
                global_best = archive(best_idx, :);
            end
        elseif t == 1
            % 第一次迭代且存档为空时，使用第一个粒子作为全局最优
            global_best = positions(1, :);
        end

        % 确保所有数组大小一致
        current_num_particles = size(positions, 1);
        
        % 向量化更新粒子速度和位置
%         r1 = rand(current_num_particles, size(positions, 2));
%         r2 = rand(current_num_particles, size(positions, 2));
        r1 = rand(current_num_particles, 1);
        r2 = rand(current_num_particles, 1);
        % 速度更新
        cognitive = c1 .* r1 .* (personal_best_positions - positions);
        social = c2 .* r2 .* (repmat(global_best, current_num_particles, 1) - positions);
        velocities = w .* velocities + cognitive + social;

        % 速度限制
        velocities = min(max(velocities, -v_max), v_max);

        % 位置更新
        new_positions = positions + velocities;

        % 边界处理（size-first：先裁剪窗口大小，再根据当前大小裁剪起始位置）
        for d = 1:dim
            % 窗口大小边界
            new_positions(:,2*d) = max(window_size_bounds(d,1), min(new_positions(:,2*d), ...
                window_size_bounds(d,2)));
            % 起始位置边界：取决于当前窗口大小
            new_positions(:,2*d-1) = max(bounds(d,1), min(new_positions(:,2*d-1), ...
                bounds(d,2) - new_positions(:,2*d)));
        end

        % 重叠检测（使用 for 循环以避免并行池内存开销）
        new_regions = zeros(current_num_particles, 2 * dim);
        new_regions(:, 1:2:end) = new_positions(:, 1:2:end);
        new_regions(:, 2:2:end) = new_positions(:, 1:2:end) + new_positions(:, 2:2:end);
        if vectorized_roi_metrics
            overlap_results = is_overlapping_batch(new_regions, occupied_regions);
        else
            overlap_results = false(current_num_particles, 1);
            for i = 1:current_num_particles
                overlap_results(i) = is_overlapping(new_regions(i,:), occupied_regions);
            end
        end

        % 更新位置和速度
        for i = 1:current_num_particles
            if ~overlap_results(i)
                positions(i,:) = new_positions(i,:);
            else
                velocities(i,:) = 0.5 * velocities(i,:);
            end
        end

        % 更新粒子结构体
        for i = 1:current_num_particles
            particles(i).position = positions(i,:);
            particles(i).velocity = velocities(i,:);
            particles(i).personal_best_position = personal_best_positions(i,:);
            particles(i).personal_best_fitness = personal_best_fitnesses(i,:);
        end

        % 计算多样性指标
        if current_num_particles > 0
            current_positions = zeros(current_num_particles, dim);
            for i = 1:current_num_particles
                for d = 1:dim
                    current_positions(i,d) = particles(i).position(2*d-1) + particles(i).position(2*d)/2;
                end
            end

            % 计算覆盖半径（添加空值保护）
            if ~isempty(current_positions)
                centroid = mean(current_positions, 1);
                distances = pdist2(current_positions, centroid);
                diversity_metrics.coverage(end+1) = max(distances);
            else
                diversity_metrics.coverage(end+1) = 0;
            end

            % 计算方差（添加空值保护）
            diversity_metrics.variance(end+1) = mean(var(current_positions, 0, 1));
        else
            % 处理无有效粒子情况
            diversity_metrics.coverage(end+1) = 0;
            diversity_metrics.variance(end+1) = 0;
        end

        % 自适应重启机制（替代原固定周期 mod(t, iterations2)）
        % 触发条件：Hypervolume 连续 patience 代无明显提升，且未停止重启
        if stagnation_counter >= patience && ~isempty(archive) && ~stop_restarting
            diagnostics.restart(t) = true;
            % 论文 S2（Merge Strategy）：
            % 1) 对 archive 进行非支配排序；
            % 2) 在第一前沿中过滤掉与现有 occupied_regions 空间重叠的候选；
            % 3) 计算剩余候选的拥挤距离；
            % 4) 从拥挤距离最大的前 k=2M 个解中均匀随机选一个，加入 occupied_regions。
            % 这里保留空间重叠过滤，是为了保证 occupied_regions 作为最终输出集时
            % 各 ROI 之间互不重叠。

            % 1) 非支配排序
            fronts = non_dominated_sorting(archive_fitnesses);
            first_front = fronts{1};

            % 2) 过滤与现有 occupied_regions 空间重叠的候选
            candidate_idx = true(length(first_front), 1);
            candidate_regions = zeros(length(first_front), 2 * dim);
            candidate_regions(:, 1:2:end) = archive(first_front, 1:2:end);
            candidate_regions(:, 2:2:end) = archive(first_front, 1:2:end) + ...
                archive(first_front, 2:2:end);
            if vectorized_roi_metrics
                candidate_idx = ~is_overlapping_batch(candidate_regions, occupied_regions);
            else
                for i = 1:length(first_front)
                    candidate_idx(i) = ~is_overlapping(candidate_regions(i,:), occupied_regions);
                end
            end

            if any(candidate_idx)
                feasible_front = first_front(candidate_idx);

                % 3) 计算拥挤距离
                crowding_distances = compute_crowding_distance(archive_fitnesses(feasible_front));

                % 4) k = 2M，取拥挤距离最大的前 k 个解
                M = num_fitness;
                k = min(2 * M, length(feasible_front));
                [~, sorted_idx] = sort(crowding_distances, 'descend');
                top_k = feasible_front(sorted_idx(1:k));

                % 均匀随机选择一个加入 occupied_regions
                best_idx = top_k(randi(k));
            else
                % 退化保护：第一前沿中无可行候选时，从整个 archive 中找第一个不重叠解
                best_idx = [];
                for i = 1:size(archive, 1)
                    region = zeros(1, 2 * dim);
                    for d = 1:dim
                        region(2 * d - 1) = archive(i, 2 * d - 1);
                        region(2 * d)   = archive(i, 2 * d - 1) + archive(i, 2 * d);
                    end
                    if ~is_overlapping(region, occupied_regions)
                        best_idx = i;
                        break;
                    end
                end

                % 若整个 archive 都无解，跳过本次加入（保留 occupied_regions 不变）
                if isempty(best_idx)
                    fprintf('迭代 %d: archive 中无可行非重叠候选，跳过本次 ROI 加入。\n', t);
                end
            end

            if ~isempty(best_idx)
                best_region = archive(best_idx, :);

                % 将 archive 中的 [start, size] 格式转换为 occupied_regions 的 [start, end] 格式
                best_region_converted = zeros(1, 2 * dim);
                for d = 1:dim
                    best_region_converted(2 * d - 1) = best_region(2 * d - 1);
                    best_region_converted(2 * d)   = best_region(2 * d - 1) + best_region(2 * d);
                end
                occupied_regions = [occupied_regions; best_region_converted];
                occupied_fitnesses = [occupied_fitnesses; archive_fitnesses(best_idx)];

                % 仅当实际加入 ROI 时才计数
                roi_collected = roi_collected + 1;
            end

            % --- 重新初始化粒子（严格多维非重叠）---
            original_particles = particles; % 备份原有粒子
            new_particles = struct('position', {}, 'velocity', {}, 'personal_best_position', {}, 'personal_best_fitness', {});

            % 参数设置
            max_attempts_reset = 50;          % 最大尝试次数
            samples_per_attempt = 500;     % 每次尝试生成的样本数
            valid_count_reset = 0;

            % 生成候选粒子并验证（size-first：先确定窗口宽度，再确定起始位置）
            for attempt = 1:max_attempts_reset
                % 生成LHS样本（在原始边界内）
                sampler = lhsdesign(samples_per_attempt, 2*dim);
                candidate_positions = zeros(samples_per_attempt, 2*dim);

                % 转换为实际位置和窗口大小
                for d = 1:dim
                    % 窗口大小范围
                    size_min = window_size_bounds(d, 1);
                    size_max = window_size_bounds(d, 2);
                    % 先生成窗口大小
                    size_val = size_min + (size_max - size_min) * sampler(:, 2*d);
                    % 起始位置范围取决于当前窗口大小：[lower, upper - size]
                    start_min = bounds(d, 1);
                    start_max = bounds(d, 2) - size_val;

                    % 映射到实际值
                    candidate_positions(:, 2*d-1) = start_min + (start_max - start_min) .* sampler(:, 2*d-1);
                    candidate_positions(:, 2*d) = size_val;
                end

                % 检查每个候选粒子
                for i = 1:size(candidate_positions, 1)
                    pos = candidate_positions(i, :);
                    region = zeros(1, 2*dim);
                    overlap_flag = false;

                    % 构造多维区域表示
                    for d = 1:dim
                        start = pos(2*d-1);
                        size_val = pos(2*d);
                        region(2*d-1) = start;
                        region(2*d) = start + size_val;

                        % 单维度边界检查（size-first 下理论上已满足，保留作为安全保护）
                        if (start < bounds(d,1)) || (start + size_val > bounds(d,2))
                            overlap_flag = true;
                            break;
                        end
                    end

                    % 全维度重叠检测（与所有占用区域比较）
                    if ~overlap_flag
                        for occ_idx = 1:size(occupied_regions, 1)
                            occ_region = occupied_regions(occ_idx, :);
                            all_dim_overlap = true;

                            % 检查每个维度是否重叠
                            for d = 1:dim
                                new_start = region(2*d-1);
                                new_end = region(2*d);
                                occ_start = occ_region(2*d-1);
                                occ_end = occ_region(2*d);

                                if (new_end <= occ_start) || (new_start >= occ_end)
                                    all_dim_overlap = false;
                                    break; % 任一维度不重叠则整体不重叠
                                end
                            end

                            % 存在任一占用区域的全维度重叠
                            if all_dim_overlap
                                overlap_flag = true;
                                break;
                            end
                        end
                    end

                    % 保留有效粒子
                    if ~overlap_flag
                        valid_count_reset = valid_count_reset + 1;
                        new_particles(valid_count_reset).position = pos;
                        new_particles(valid_count_reset).velocity = -v_max + 2*v_max.*rand(1, 2*dim);
                        new_particles(valid_count_reset).personal_best_position = pos;
                        new_particles(valid_count_reset).personal_best_fitness = -inf(1, num_fitness);

                        if valid_count_reset >= num_particles
                            break;
                        end
                    end
                end

                % 检查是否完成
                if valid_count_reset >= num_particles
                    break;
                end
            end
            
            % 更新粒子群
            if valid_count_reset > 0
                particles = new_particles(1:valid_count_reset);
                num_particles = valid_count_reset;
                
                % 更新矩阵变量
                positions = vertcat(particles.position);
                velocities = vertcat(particles.velocity);
                personal_best_positions = vertcat(particles.personal_best_position);
                personal_best_fitnesses = vertcat(particles.personal_best_fitness);
                fitnesses = zeros(num_particles, num_fitness);
            else
                particles = original_particles;
                % 如果重置失败，保持原有粒子数量
                num_particles = length(original_particles);
            end
            
            % 重置 PSO 参数（保留 archive，不清空）
            % 每次重启视为新的 ROI 优化，重置参数调度与 Hypervolume 基准
            iter_since_restart = 0;
            last_hypervolume = 0;

            % 重置停滞计数器
            stagnation_counter = 0;

            if ~isempty(best_idx)
                % 成功加入 ROI，重置空白重启计数
                blank_restart_counter = 0;
                fprintf('迭代 %d: Hypervolume 停滞 %d 代，触发自适应重启，成功加入第 %d 个 ROI\n', ...
                    t, patience, roi_collected);
            else
                % 未找到可行 ROI
                blank_restart_counter = blank_restart_counter + 1;
                fprintf('迭代 %d: Hypervolume 停滞 %d 代，触发自适应重启，但未找到非重叠候选（连续 %d 次）\n', ...
                    t, patience, blank_restart_counter);
                if blank_restart_counter >= max_blank_restarts
                    stop_restarting = true;
                    fprintf('连续 %d 次未找到新 ROI，停止后续重启，继续优化当前 ROI。\n', max_blank_restarts);
                end
            end
        end

        diagnostics.roi_collected(t) = roi_collected;
        elapsed = toc;
        fprintf('迭代 %d/%d, 运行时间: %.2f 秒\n', t, iterations1, elapsed);
    end

    %%% 结果输出部分（保持原有格式）%%%
    % 空 archive / occupied_regions 保护：若整个优化过程未找到任何非支配解，
    % 则添加一个覆盖整个输入空间的默认 ROI，避免后续 cell2mat 与 table 构造失败。
    if isempty(archive) || isempty(archive_fitnesses)
        warning('mopso_adaptive_elitism_crowding: Archive 为空，添加默认全局 ROI。');
        default_pos = zeros(1, 2*dim);
        for d = 1:dim
            default_pos(2*d-1) = bounds(d,1);
            default_pos(2*d) = bounds(d,2) - bounds(d,1);
        end
        archive = default_pos;
        archive_fitnesses = {zeros(1, num_fitness)};
    end
    
    if isempty(occupied_regions) || isempty(occupied_fitnesses)
        warning('mopso_adaptive_elitism_crowding: occupied_regions 为空，添加默认全局 ROI。');
        default_region = zeros(1, 2*dim);
        for d = 1:dim
            default_region(2*d-1) = bounds(d,1);
            default_region(2*d) = bounds(d,2);  % 使用 end 而非 size，保持 [start, end] 格式
        end
        occupied_regions = default_region;
        occupied_fitnesses = {zeros(1, num_fitness)};
    end
    
    % 调整解的位置
    adjusted_archive = cell(size(archive,1),1);
    for i = 1:size(archive,1)
        adjusted = zeros(1, dim*2);
        for d = 1:dim
            adjusted(2*d-1) = archive(i,2*d-1);
            adjusted(2*d) = archive(i,2*d);
        end
        adjusted_archive{i} = adjusted;
    end

    % 生成完整Pareto表格
    pareto_table = table();
    for d = 1:dim
        pareto_table.(['x',num2str(d),'_start']) = cellfun(@(x) x(2*d-1), adjusted_archive);
        pareto_table.(['x',num2str(d),'_end']) = cellfun(@(x) x(2*d-1)+x(2*d), adjusted_archive);
    end
    fitness_matrix = cell2mat(archive_fitnesses);
    pareto_table.IL = fitness_matrix(:,1);
    pareto_table.G = fitness_matrix(:,2);
    if num_fitness >= 3
        pareto_table.U = fitness_matrix(:,3);
    end
%     pareto_table.KL = fitness_matrix(:,3);
%     pareto_table.NE = int32(fitness_matrix(:,4));

    % 生成Top10表格
    [~, sorted_idx] = sort(fitness_matrix(:,1), 'descend');
    top10_idx = sorted_idx(1:min(10,length(sorted_idx)));
    top10_table = pareto_table(top10_idx,:);

    % 输出到Excel
    % occupied_regions 是按论文 S2 逐步收集的最终 ROI 输出集 A'
    % 每次重置已用拥挤距离前 2M 随机选择并保证空间不重叠
    occupied_table = table();
    for d = 1:dim
        occupied_table.(['x',num2str(d),'_start']) = occupied_regions(:, 2*d-1);
        occupied_table.(['x',num2str(d),'_end'])   = occupied_regions(:, 2*d);
    end
    occupied_fitness_matrix = cell2mat(occupied_fitnesses);
    occupied_table.IL = occupied_fitness_matrix(:,1);
    occupied_table.G  = occupied_fitness_matrix(:,2);
    if num_fitness >= 3
        occupied_table.U = occupied_fitness_matrix(:,3);
    end
%     occupied_table.KL = -occupied_fitness_matrix(:,3);
%     occupied_table.NE = int32(occupied_fitness_matrix(:,4));
%     writetable(occupied_table, location,'Sheet', 'Final results');
%     writetable(pareto_table, location, 'Sheet', 'time_Pareto_Front');
%     writetable(top10_table, location, 'Sheet', 'Top 10 Pareto');
    
%     % 参数动态输出 - 确保所有列长度一致
%     % 使用实际完成的迭代次数而不是预分配的长度
%     iteration_indices = (1:actual_iterations)';
%     inertia_weights = parameter_metrics.inertia_weight(1:actual_iterations);
%     cognitive_factors = parameter_metrics.cognitive_factor(1:actual_iterations);
%     social_factors = parameter_metrics.social_factor(1:actual_iterations);
%     
%     parameter_table = table(iteration_indices, inertia_weights, ...
%         cognitive_factors, social_factors, ...
%         'VariableNames', {'Iteration','Inertia_Weight','Cognitive_Factor','Social_Factor'});
%     writetable(parameter_table, location, 'Sheet', 'Parameter Dynamics');
end
