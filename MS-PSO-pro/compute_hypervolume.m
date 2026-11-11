function hv = compute_hypervolume(front, ref, n_samples, direction)
    % compute_hypervolume  计算 Pareto 前沿的 Hypervolume 指标
    %
    % 输入：
    %   front      - N x M 矩阵，每行是一个非支配解的目标向量
    %   ref        - 1 x M 参考点
    %                最小化问题：ref 比 front 中所有值都大（worst）
    %                最大化问题：ref 比 front 中所有值都小（worst）
    %   n_samples  - 蒙特卡洛采样点数（高维时使用，默认 20000）
    %   direction  - 'min' 或 'max'，表示最小化还是最大化问题（默认 'min'）
    %
    % 输出：
    %   hv         - Hypervolume 值

    if nargin < 3 || isempty(n_samples)
        n_samples = 20000;
    end
    if nargin < 4 || isempty(direction)
        direction = 'min';
    end

    if isempty(front)
        hv = 0;
        return;
    end

    % 统一转换为最小化问题处理
    if strcmpi(direction, 'max')
        front = -front;
        ref = -ref;
    end

    [n, m] = size(front);

    % 过滤掉被参考点支配的解（这些解对 hypervolume 无贡献）
    valid = all(front <= repmat(ref, n, 1), 2);
    front = front(valid, :);

    if isempty(front)
        hv = 0;
        return;
    end

    % 二维精确计算
    if m == 2
        hv = hypervolume_2d(front, ref);
        return;
    end

    % 高维蒙特卡洛近似
    hv = hypervolume_mc(front, ref, n_samples);
end


function hv = hypervolume_2d(front, ref)
    % 二维精确计算：按第一个目标升序，累加矩形面积
    front = sortrows(front, 1);

    hv = 0;
    prev_y = ref(2);

    for i = 1:size(front, 1)
        width = ref(1) - front(i, 1);
        height = prev_y - front(i, 2);
        if width > 0 && height > 0
            hv = hv + width * height;
        end
        prev_y = front(i, 2);
    end
end


function hv = hypervolume_mc(front, ref, n_samples)
    % 高维蒙特卡洛近似
    [~, m] = size(front);

    % 采样空间为 [min(front), ref]
    lb = min(front, [], 1);
    ub = ref;

    box_size = ub - lb;
    if any(box_size <= 0)
        hv = 0;
        return;
    end

    box_volume = prod(box_size);

    % 均匀随机采样
    samples = lb + rand(n_samples, m) .* box_size;

    % 向量化判断每个样本是否被至少一个 front 中的点支配
    samples_3d = reshape(samples, n_samples, 1, m);
    front_3d = permute(front, [3, 1, 2]);  % 1 x n x m

    dominated_matrix = all(samples_3d >= front_3d, 3);  % n_samples x n
    dominated_count = sum(any(dominated_matrix, 2));

    hv = (dominated_count / n_samples) * box_volume;
end
