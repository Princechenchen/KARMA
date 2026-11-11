function local_minima = local_optimization(func, initial_points, method)
    if nargin < 3
        method = 'fminunc';  % Default optimization method
    end
    local_minima = [];
    for i = 1:size(initial_points, 1)
        point = initial_points(i, :);
        try
            if strcmpi(method, 'fminunc')
                % 使用 optimoptions 配置 fminunc
                options = optimoptions('fminunc', ...
                    'Algorithm', 'quasi-newton', ...  % 使用拟牛顿法
                    'Display', 'none', ...
                    'MaxIterations', 1000, ...        % 增加最大迭代次数
                    'OptimalityTolerance', 1e-8, ...  % 调整优化容差
                    'StepTolerance', 1e-8);
                [x, fval, exitflag] = fminunc(func, point, options);
            else
                % 其他优化方法（如 fminsearch）
                options = optimset('Display', 'none', 'MaxIter', 1000);
                [x, fval, exitflag] = fminsearch(func, point, options);
            end
            
            % 检查优化是否成功
            if exitflag > 0
                local_minima = [local_minima; x, fval];
            end
        catch ME
            % 捕获并显示警告，但继续执行其他初始点
            a=1;
        end
    end
end
