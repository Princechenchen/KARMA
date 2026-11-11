function fn = test2(x, model)%x输入是矩阵
    fn = zeros(size(x, 1), 1); 
    
    % Loop through each row of x
    for row = 1:size(x, 1)
        n = length(x(row, :));  % 获取当前行的长度，即n的值
        sum_cos4 = 0;           % 初始化sum(cos^4(x_i))
        prod_cos2 = 1;          % 初始化prod(cos^2(x_i))
        sum_i_x2 = 0;           % 初始化sum(i * x_i^2)
        
        % 计算公式中的各个部分
        for i = 1:n
            sum_cos4 = sum_cos4 + cos(x(row, i))^4;
            prod_cos2 = prod_cos2 * cos(x(row, i))^2;
            sum_i_x2 = sum_i_x2 + i * x(row, i)^2;
        end
        
        % 计算分母中的平方根
        sqrt_sum_i_x2 = sqrt(sum_i_x2);
        
        % 计算整个函数的值
        fn(row) = -abs(sum_cos4 - 2 * prod_cos2) / sqrt_sum_i_x2;
    end
end