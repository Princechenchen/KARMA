function occupied_table = mopso_adaptive_elitism_crowdingi(dim, bounds, window_size_bounds, ...
    model, num_particles, iterations1, iterations2, num_samples, elitism_size, model_num)
    
    % 初始化参数记录结构
    parameter_metrics = struct('inertia_weight', [], 'cognitive_factor', [], 'social_factor', []);
    
    % 新增：初始化占用区域和适应度存储
    occupied_regions = [];
    occupied_fitnesses = cell(0);
    max_archive_size = 20; % 最大存档数量
    num_fitness = 2; %1+size(model.ClassPos,2)

    % 速度限制计算
    v_max = [];
    for d = 1:dim
        v_max = [v_max, (bounds(d,2)-bounds(d,1))*0.2];
        v_max = [v_max, (window_size_bounds(d,2)-window_size_bounds(d,1))*0.2];
    end

    % 边界向量准备
    l_bounds = []; u_bounds = [];
    for d = 1:dim
        l_bounds = [l_bounds, bounds(d,1), window_size_bounds(d,1)];
        u_bounds = [u_bounds, bounds(d,2)-window_size_bounds(d,2), window_size_bounds(d,2)];
    end

    % 初始化粒子群（带重叠检测）
    max_attempts = 10;  % 最大尝试次数
    valid_count = 0;
    total_generated = 0;
    particles = struct('position', {}, 'velocity', {}, 'personal_best_position', {}, 'personal_best_fitness', {});

    for attempt = 1:max_attempts
        % 每次生成双倍样本（可根据需要调整倍数）
        sampler = lhsdesign(num_particles*2, 2*dim);

        % 处理当前批次的样本
        for i = 1:size(sampler,1)
            % 位置生成
            pos = sampler(i,:) .* (u_bounds - l_bounds) + l_bounds;

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
                particles(valid_count).personal_best_fitness = inf(1,num_fitness);

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

    % 动态参数设置（保持原有tan函数策略）
    w0 = 0.72; wK = 1.17;
    c1_0 = 1.54; c1_K = 2.07;
    c2_0 = 1.31; c2_K = 2.86;
    delta = 0.1;

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
    
    % 主优化循环
    for t = 1:iterations1
        tic;
        actual_iterations = actual_iterations + 1;

        % 动态参数计算
% 动态参数计算
k_iter = t;
K = iterations1;
w = (w0 - wK) * tan((1 - k_iter/K) + delta) + wK;
c1 = (c1_0 - c1_K) * tan(k_iter/K + delta) + c1_K;
c2 = (c2_0 - c2_K) * tan((1 - k_iter/K) + delta) + c2_K;

% 记录参数
parameter_metrics.inertia_weight(t) = w;
parameter_metrics.cognitive_factor(t) = c1;
parameter_metrics.social_factor(t) = c2;

% 在parfor循环之前，显式创建所有需要的临时变量
temp_model = model;
% temp_kde_q_ = kde_q_;
temp_num_samples = num_samples;
temp_dim = dim;
temp_model_num = model_num;
% temp_all_extrema = all_extrema;
temp_occupied_regions = occupied_regions; % 创建常量副本

% 并行评估适应度
parfor i = 1:num_particles
    fitnesses(i,:) = evaluate_fitnessi(positions(i,:), temp_model, ...
        temp_num_samples, temp_dim, temp_model_num, temp_occupied_regions);
end

        % 更新个体最优
        for i = 1:num_particles
            if dominates(fitnesses(i,:), personal_best_fitnesses(i,:))
                personal_best_positions(i,:) = positions(i,:);
                personal_best_fitnesses(i,:) = fitnesses(i,:);
            end

            % 更新存档
            [archive, archive_fitnesses] = update_archive(archive, archive_fitnesses, ...
                positions(i,:), fitnesses(i,:), max_archive_size);
        end

        % 非支配排序和拥挤度计算（减少频率）
        if mod(t, 5) == 0 && ~isempty(archive_fitnesses)  % 每5次迭代执行一次
            fronts = non_dominated_sorting(archive_fitnesses);
            first_front = fronts{1};
            crowding_distances = compute_crowding_distance(archive_fitnesses(first_front,:));

            % 精英选择
            [~, sorted_idx] = sort(crowding_distances, 'descend');
            elites = first_front(sorted_idx(1:min(elitism_size,length(first_front))));

            % 全局最优选择
            if ~isempty(elites)
                global_best = archive(elites(1),:);
            else
                global_best = mean(archive,1);
            end
        elseif isempty(archive_fitnesses) && t == 1
            % 第一次迭代且存档为空时，使用第一个粒子作为全局最优
            global_best = positions(1,:);
        end

        % 确保所有数组大小一致
        current_num_particles = size(positions, 1);
        
        % 向量化更新粒子速度和位置
        r1 = rand(current_num_particles, size(positions, 2));
        r2 = rand(current_num_particles, size(positions, 2));

        % 速度更新
        cognitive = c1 .* r1 .* (personal_best_positions - positions);
        social = c2 .* r2 .* (repmat(global_best, current_num_particles, 1) - positions);
        velocities = w .* velocities + cognitive + social;

        % 速度限制
        velocities = min(max(velocities, -v_max), v_max);

        % 位置更新
        new_positions = positions + velocities;

        % 边界处理
        for d = 1:dim
            % 起始位置边界
            new_positions(:,2*d-1) = max(bounds(d,1), min(new_positions(:,2*d-1), ...
                bounds(d,2) - window_size_bounds(d,2)));
            % 窗口大小边界
            new_positions(:,2*d) = max(window_size_bounds(d,1), min(new_positions(:,2*d), ...
                bounds(d,2) - new_positions(:,2*d-1)));
        end

        % 重叠检测
        overlap_results = false(current_num_particles, 1);
        parfor i = 1:current_num_particles
            region = zeros(1, dim*2);
            for d = 1:dim
                region(2*d-1) = new_positions(i,2*d-1);
                region(2*d) = new_positions(i,2*d-1) + new_positions(i,2*d);
            end
            overlap_results(i) = is_overlapping(region, occupied_regions);
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

        % 周期重置机制（每iterations2次迭代）
        if mod(t, iterations2) == 0 && ~isempty(archive)
            % 选择最优解加入占用区域
            [~, best_idx] = max(cellfun(@(x) x(1), archive_fitnesses));
            best_region = archive(best_idx, :);
            occupied_regions = [occupied_regions; best_region];
            occupied_fitnesses = [occupied_fitnesses; archive_fitnesses(best_idx)];

            % --- 重新初始化粒子（严格多维非重叠）---
            original_particles = particles; % 备份原有粒子
            new_particles = struct('position', {}, 'velocity', {}, 'personal_best_position', {}, 'personal_best_fitness', {});

            % 参数设置
            max_attempts_reset = 50;          % 最大尝试次数
            samples_per_attempt = 500;     % 每次尝试生成的样本数
            valid_count_reset = 0;

            % 生成候选粒子并验证
            for attempt = 1:max_attempts_reset
                % 生成LHS样本（在原始边界内）
                sampler = lhsdesign(samples_per_attempt, 2*dim);
                candidate_positions = zeros(samples_per_attempt, 2*dim);

                % 转换为实际位置和窗口大小
                for d = 1:dim
                    % 起始点范围
                    start_min = bounds(d, 1);
                    start_max = bounds(d, 2) - window_size_bounds(d, 1);

                    % 窗口大小范围
                    size_min = window_size_bounds(d, 1);
                    size_max = window_size_bounds(d, 2);

                    % 映射到实际值
                    candidate_positions(:, 2*d-1) = start_min + (start_max - start_min) * sampler(:, 2*d-1);
                    candidate_positions(:, 2*d) = size_min + (size_max - size_min) * sampler(:, 2*d);
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

                        % 单维度边界检查
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
                        new_particles(valid_count_reset).personal_best_fitness = inf(1, num_fitness);

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
            
            % 清空存档
            archive = [];
            archive_fitnesses = cell(0);
            w0 = 0.72; wK = 1.17;
            c1_0 = 1.54; c1_K = 2.07;
            c2_0 = 1.31; c2_K = 2.86;
            delta = 0.1;
        end

        elapsed = toc;
        fprintf('迭代 %d/%d, 运行时间: %.2f 秒\n', t, iterations1, elapsed);
    end

    %%% 结果输出部分（保持原有格式）%%%
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
%     pareto_table.KL = fitness_matrix(:,3);
%     pareto_table.NE = int32(fitness_matrix(:,4));

    % 生成Top10表格
    [~, sorted_idx] = sort(fitness_matrix(:,1), 'descend');
    top10_idx = sorted_idx(1:min(10,length(sorted_idx)));
    top10_table = pareto_table(top10_idx,:);

    % 输出到Excel
    occupied_table = table();
    for d = 1:dim
        occupied_table.(['x',num2str(d),'_start']) = occupied_regions(:,2*d-1);
        occupied_table.(['x',num2str(d),'_end']) = occupied_regions(:,2*d-1)+occupied_regions(:,2*d);
    end
    occupied_fitness_matrix = cell2mat(occupied_fitnesses);
    occupied_table.IL = occupied_fitness_matrix(:,1);
    occupied_table.G = occupied_fitness_matrix(:,2);
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