function [selected_rois, selected_fitnesses] = select_diverse_rois(archive, archive_fitnesses, RoI_num, dim)
    % select_diverse_rois  从累积 archive 中精选 RoI_num 个高质量、空间分散的 ROI
    %
    % 输入：
    %   archive          - N x 2D 矩阵，每行 [start1, size1, start2, size2, ...]
    %   archive_fitnesses - N x 1 cell array，每个 cell 为 [IL, G, U] 或 [IL, G]
    %   RoI_num          - 目标 ROI 数量
    %   dim              - 输入维度
    % 输出：
    %   selected_rois      - K x 2D 矩阵，K <= RoI_num
    %   selected_fitnesses - K x 1 cell array

    selected_rois = [];
    selected_fitnesses = {};

    n = size(archive, 1);
    if n == 0 || RoI_num <= 0
        return;
    end

    % 构造区域表示 [start1, end1, start2, end2, ...]
    regions = zeros(n, 2 * dim);
    for i = 1:n
        for d = 1:dim
            regions(i, 2 * d - 1) = archive(i, 2 * d - 1);
            regions(i, 2 * d)   = archive(i, 2 * d - 1) + archive(i, 2 * d);
        end
    end

    % 对所有目标做 min-max 归一化，等权计算质量分
    fit_mat = cell2mat(archive_fitnesses);
    fit_min = min(fit_mat, [], 1);
    fit_max = max(fit_mat, [], 1);
    fit_range = fit_max - fit_min;
    fit_range(fit_range == 0) = 1;  % 防止除零
    norm_fit = (fit_mat - fit_min) ./ fit_range;
    quality = mean(norm_fit, 2);    % 等权：2 目标各占 1/2，3 目标各占 1/3

    lambda = 0.5;  % 多样性惩罚系数
    selected = false(n, 1);
    selected_count = 0;

    % 贪心选择：每次选质量分最高且与已选区域重叠惩罚最小的候选
    while selected_count < RoI_num
        scores = quality;
        for i = 1:n
            if selected(i)
                continue;
            end
            penalty = 0;
            for j = find(selected)'
                penalty = penalty + compute_overlap_ratio(regions(i, :), regions(j, :), dim);
            end
            scores(i) = scores(i) - lambda * penalty;
        end
        scores(selected) = -inf;

        [best_score, best_idx] = max(scores);
        if ~isfinite(best_score)
            break;  % 没有更多可行候选
        end

        selected(best_idx) = true;
        selected_count = selected_count + 1;
    end

    selected_idx = find(selected);
    selected_rois = archive(selected_idx, :);
    selected_fitnesses = archive_fitnesses(selected_idx);
end
