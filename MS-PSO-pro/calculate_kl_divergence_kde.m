% function kl_div = calculate_kl_divergence_kde(rect_samples, kde_q_, bandwidth)
%     n_dimensions = size(rect_samples, 2);
% 
%     % 使用 mvksdensity 拟合样本点的概率密度 P(x)
%     p = mvksdensity(rect_samples, rect_samples, 'Bandwidth', bandwidth);
% 
%     % 计算样本点上 Q(x) 的概率密度
%     q = kde_q_(rect_samples);
% 
%     % 避免在计算中出现零概率
%     epsilon = 1e-10;
%     p = p + epsilon;
%     q = q + epsilon;
% 
%     % 重新归一化（确保概率密度）
%     p = p / sum(p);
%     q = q / sum(q);
% 
%     % 计算KL散度 D(P || Q) = Σ P(x) log(P(x)/Q(x))
%     kl_div = sum(p .* (log(p) - log(q)));
% end

function kl_div = calculate_kl_divergence_kde(test_samples, kde_q_func, bandwidth)
    if isempty(test_samples)
        kl_div = 0.0;
        return
    end
    
    % 确保输入为N×D矩阵
    if isvector(test_samples)
        test_samples = test_samples(:); 
    end
    
    % 计算P的密度（当前样本分布）
    [n, d] = size(test_samples);
    p = mvksdensity(test_samples, test_samples, ...
                   'Bandwidth', bandwidth, ...
                   'Kernel', 'normal');
    
    % 计算Q的密度（参考分布）
    log_q = kde_q_func(test_samples); % 使用预定义的KDE函数
    q = exp(log_q);
    
    % 数值稳定性处理
    epsilon = 1e-10;
    p = max(p, epsilon);
    q = max(q, epsilon);
    
    % 计算KL散度
    kl_div = mean(log(p ./ q));
end
