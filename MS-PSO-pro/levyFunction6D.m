function f = levyFunction6D(x)
    f = 0;
    
    % 计算w_i
    w = 1 + (x - 1) / 4;
    
    % 计算函数值
    f = sin(pi * w(1))^2;
    for i = 2:6
        if i < 6
            f = f + (w(i) - 1)^2 * (1 + 10 * sin(pi * w(i) + 1)^2);
        else
            f = f + (w(i) - 1)^2 * (1 + sin(2 * pi * w(i))^2);
        end
    end
end