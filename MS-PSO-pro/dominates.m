function is_dominated = dominates(fitness1, fitness2)
    % 确保 fitness1 和 fitness2 是数值数组
    if iscell(fitness1)
        fitness1 = cell2mat(fitness1);
    end
    if iscell(fitness2)
        fitness2 = cell2mat(fitness2);
    end
    
    % 检查支配关系
    is_dominated = all(fitness1 >= fitness2) && any(fitness1 > fitness2);
end