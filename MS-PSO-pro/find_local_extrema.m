function final_minima = find_local_extrema(func, dim, num_points, lb, ub, method, tol)
    % 初始化候选点
    init_points = random_initialization(dim, num_points, lb, ub);
    
    % 执行局部优化
    candidate_minima = {};
    for i = 1:size(init_points, 1)
        opts = optimoptions('fminunc',...
            'Algorithm', 'quasi-newton',...
            'HessUpdate', 'bfgs',...
            'Display', 'off');
        
        [x_opt, fval, exitflag] = fminunc(func, init_points(i,:), opts);
        if exitflag > 0
            candidate_minima{end+1} = {x_opt, fval};
        end
    end
    
    % 去除重复极值点
    final_minima = remove_duplicates(candidate_minima, tol);
end