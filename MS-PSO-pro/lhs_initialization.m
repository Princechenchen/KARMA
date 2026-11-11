function points = lhs_initialization(dim, num_points, lower_bound, upper_bound)
    % 生成拉丁超立方样本
    points = lhsdesign(num_points, dim, 'criterion', 'maximin', 'iterations', 1000);
    
    % 将样本点缩放到指定的范围
    points = (upper_bound - lower_bound) .* points + lower_bound;
end