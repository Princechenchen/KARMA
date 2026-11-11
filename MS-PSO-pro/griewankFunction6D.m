function f = griewankFunction6D(x)

    f = 1;
    
    % 计算平方和
    sumSquares = sum(x.^2);
    
    % 计算余弦乘积
    productCos = 1;
    for i = 1:6
        productCos = productCos * cos(x(i) / sqrt(i));
    end
    
    % 计算函数值
    f = f + (1/4000) * sumSquares - productCos;
end