function G_max = compute_gradient_metric(samples, sampledY) %多响应梯度平均值
    num_steps = size(samples, 1);
    if num_steps < 2
        G_max = 0.0;
        return
    end
    dimensions = size(samples, 2);
    ranges = max(samples) - min(samples);
    s = sum(ranges);
    if s == 0
        s = 1e-12;
    end
    fitness = sampledY;
    f_range = max(fitness) - min(fitness);
%     if f_range == 0
%         f_range = 1e-12;
%     end
    f_range(f_range==0) = 1e-12;
%     Gradient = zeros(num_steps-1, 1);
    for i = 1:num_steps-1
        delta_norm = norm(samples(i+1, :) - samples(i, :));
        if delta_norm == 0
            Gradient(i,:) = 0;
        else
            Gradient(i,:) = (fitness(i+1,:) - fitness(i,:)) ./ f_range / (delta_norm / s);
        end
    end
    G_max = max(mean(abs(Gradient),2));
end