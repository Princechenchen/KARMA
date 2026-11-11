function vol = compute_union_volume(archive, dim, bounds, n_samples)
    % compute_union_volume  用蒙特卡洛方法估计 archive 中所有 ROI 区域的并集体积
    %
    % 输入：
    %   archive  - N x (2*dim) 矩阵，每行表示一个 ROI 区域，
    %              第 (2d-1) 列为第 d 维起始位置，第 (2d) 列为第 d 维宽度
    %   dim      - 空间维度
    %   bounds   - dim x 2 矩阵，bounds(d,:) 为第 d 维的 [下界, 上界]
    %   n_samples- 蒙特卡洛采样点数（默认 10000）
    %
    % 输出：
    %   vol      - 所有 ROI 区域并集的体积估计值

    if nargin < 4 || isempty(n_samples)
        n_samples = 10000;
    end

    if isempty(archive)
        vol = 0;
        return;
    end

    n = size(archive, 1);

    % 将 [start, size] 格式转换为 [start, end] 格式
    regions = zeros(n, 2 * dim);
    for i = 1:n
        for d = 1:dim
            start_val = archive(i, 2*d - 1);
            size_val = archive(i, 2*d);
            regions(i, 2*d - 1) = start_val;
            regions(i, 2*d) = start_val + size_val;
        end
    end

    % 计算所有 ROI 区域的外接盒子，并裁剪到问题边界内
    lb = min(regions(:, 1:2:end), [], 1);
    ub = max(regions(:, 2:2:end), [], 1);
    lb = max(lb, bounds(:, 1)');
    ub = min(ub, bounds(:, 2)');

    box_size = ub - lb;
    if any(box_size <= 0)
        vol = 0;
        return;
    end

    box_volume = prod(box_size);

    % 在 bounding box 内均匀随机采样
    samples = lb + rand(n_samples, dim) .* box_size;

    % 向量化判断每个样本是否被至少一个 ROI 覆盖
    starts = regions(:, 1:2:end);  % n x dim
    ends = regions(:, 2:2:end);    % n x dim

    samples_3d = reshape(samples, n_samples, 1, dim);
    starts_3d = permute(starts, [3, 1, 2]);  % 1 x n x dim
    ends_3d = permute(ends, [3, 1, 2]);      % 1 x n x dim

    inside = (samples_3d >= starts_3d) & (samples_3d <= ends_3d);
    inside_all = all(inside, 3);             % n_samples x n
    covered_count = sum(any(inside_all, 2));

    vol = (covered_count / n_samples) * box_volume;
end
