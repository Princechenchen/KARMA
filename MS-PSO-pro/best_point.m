function best = best_point(samples, sampledY)
    % best_point  按方案 B：列归一化后取离理想点最近的样本作为参考点
    %
    % 默认按最小化方向处理：理想点为归一化响应空间的原点。
    % 若目标为最大化，可在调用时对 sampledY 取负号，即
    %   best_point(samples, -sampledY)

    n = size(samples, 1);
    if n == 0
        best = [];
        return;
    end

    % 列归一化到 [0,1]，消除多响应尺度差异
    y_min = min(sampledY, [], 1);
    y_max = max(sampledY, [], 1);
    y_range = y_max - y_min;
    y_range(y_range == 0) = 1;
    Y_norm = (sampledY - y_min) ./ y_range;

    % 最小化方向：理想点为原点
    ideal = zeros(1, size(sampledY, 2));
    dist = sqrt(sum((Y_norm - ideal).^2, 2));

    [~, idx] = min(dist);
    best = samples(idx, :);
end
