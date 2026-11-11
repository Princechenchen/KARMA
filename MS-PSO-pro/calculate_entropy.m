function IL_ns = calculate_entropy(x, sampledY, erro)
    n = size(x, 1);
    if n == 0
        IL_ns = 0.0;
        return;
    end
    x_star = best_point(x, sampledY);

    % 对响应值做列归一化，消除连续响应与分类响应之间的尺度差异
    y_min = min(sampledY, [], 1);
    y_max = max(sampledY, [], 1);
    y_range = y_max - y_min;
    y_range(y_range == 0) = 1;
    f = (sampledY - y_min) ./ y_range;

    vp = zeros(n, n);
    for i = 1:n
        for j = i+1:n
            fi = f(i, :);
            fj = f(j, :);
            vp(i, j)=0; dim = size(fi,2); %%多响应累加平均处理
            for d=1:dim
            if fi(d) < fj(d) - erro
                vp(i, j) = vp(i, j)+1;
            elseif abs(fi(d) - fj(d)) < erro
                vp(i, j) = vp(i, j)+0.5;
            else
                vp(i, j) = vp(i, j)+0;
            end
            end
            vp(i, j) = vp(i, j)/dim;
        end
    end
    f_r = @(point) sum(point.^2) - 2 * dot(point, x_star) + sum(x_star.^2);
    v_r = arrayfun(@(k) f_r(x(k, :)), 1:n)';
    vr = zeros(n, n);
    for i = 1:n
        for j = i+1:n
            if v_r(i) < v_r(j) - erro
                vr(i, j) = 1;
            elseif abs(v_r(i) - v_r(j)) < erro
                vr(i, j) = 0.5;
            else
                vr(i, j) = 0;
            end
        end
    end
    sum_abs = sum(abs(vp - vr), 'all');
    IL_ns = sum_abs / (n*(n+1)/2);
end