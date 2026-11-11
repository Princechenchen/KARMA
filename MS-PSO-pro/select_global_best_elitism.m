function best = select_global_best_elitism(archive, fitnesses)
    % 确保 fitnesses 是元胞数组
    if ~iscell(fitnesses)
        error('fitnesses 必须是元胞数组');
    end

    % 非支配排序
    fronts = non_dominated_sorting(fitnesses);
    first_front = fronts{1};

    % 计算拥挤度
    crowding_distances = compute_crowding_distance(fitnesses, first_front);

    % 按拥挤度排序
    [~, sorted_indices] = sort(crowding_distances, 'descend');

    % 选择拥挤度最大的解
    if iscell(archive)
        best = archive{sorted_indices(1)};  
    else
        best = archive(sorted_indices(1), :);  
    end
end