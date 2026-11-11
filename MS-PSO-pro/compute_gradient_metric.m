function G_max = compute_gradient_metric(samples, sampledY) %多响应梯度平均值
    num_steps = size(samples, 1);
    if num_steps < 2
        G_max = 0.0;
        return
    end
    ranges = max(samples) - min(samples);
    s = sum(ranges);
    if s == 0
        s = 1e-12;
    end
    fitness = sampledY;
    f_range = max(fitness) - min(fitness);
    f_range(f_range==0) = 1e-12;

    % 计算每个样本在输入空间中的最近邻（排除自身）
    D = zeros(num_steps, num_steps);
    for i = 1:num_steps
        D(:, i) = sqrt(sum((samples - samples(i, :)).^2, 2));
    end
    D(1:num_steps+1:end) = inf;
    [min_dist, nn_idx] = min(D, [], 2);

    Gradient = zeros(num_steps, size(fitness, 2));
    for i = 1:num_steps
        j = nn_idx(i);
        delta_norm = min_dist(i);
        if delta_norm == 0
            Gradient(i, :) = 0;
        else
            Gradient(i, :) = abs(fitness(j, :) - fitness(i, :)) ./ f_range / (delta_norm / s);
        end
    end

    % 与原来保持一致：每个样本内对响应维度取平均，再取所有样本中的最大值
    G_max = max(mean(abs(Gradient), 2));
end
