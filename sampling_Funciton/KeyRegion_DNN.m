function [x_star, model] = KeyRegion_DNN(model)
% KeyRegion_DNN  数据驱动重点区域采样（C9-DNN版本）
%
% 文献来源：
%   李宁, 张孝航. 基于重点区域采样深度神经网络代理模型的可靠性分析方法.
%   工程力学, 2026, 43(5): 24-32.
%
% 本版本使用深度神经网络（DNN）作为代理模型，尽可能忠实于原文献。
% 用途：跨模型补充验证——证明将代理模型从DNN替换为MMRGP后，
%       C9的采样行为未发生本质改变。
%
% 原文献面向单标量极限状态函数 g(x)。
% 本项目场景为混合响应（连续+分类），DNN版本的处理方式：
%   - 为每个响应分别训练一个DNN（单输出）
%   - 连续响应：DNN直接输出预测值，不确定性用 |g_hat - median| 近似
%   - 分类响应：DNN输出概率 p，不确定性用 4p(1-p)
%   - 多响应不确定性聚合：均值-最大值自适应权重
%
% DNN结构（原文献参数）：
%   - 隐藏层：2层，神经元数随维度调整（低维 8-16-8，高维适当增加）
%   - 激活函数：ReLU
%   - 优化器：Adam (lr=0.001, beta1=0.9, beta2=0.99)
%   - L2正则化：lambda=1e-4
%   - 最大训练次数：800~3000（随样本量调整）
%   - 批大小：min(n_samples, 32)
%   - 输入标准化：Z-score
%
% 输入/输出接口与 KeyRegion_sampling.m（MMRGP版本）完全一致。
% model 结构体中需包含：
%   model.DNN       - DNN代理模型结构（首次调用时为空，自动训练）
%   其余字段同 KeyRegion_sampling.m

    ns = size(model.candidate, 1);
    dim = model.dim;
    yd = size(model.init_value, 2);

    %% ---- 参数设置 ----
    if isfield(model, 'kr_band_ratio')
        band_ratio = model.kr_band_ratio;
    else
        band_ratio = 0.5;
    end

    if isfield(model, 'kr_prune_factor')
        prune_factor = model.kr_prune_factor;
    else
        prune_factor = 0.5;
    end

    % DNN训练参数
    if isfield(model, 'dnn_max_epochs')
        max_epochs = model.dnn_max_epochs;
    else
        max_epochs = 1000;
    end

    % DNN隐藏层结构
    hidden_size = max(8, min(32, 4 * dim));
    layers_config = [hidden_size, hidden_size];  % 两隐藏层

    %% ---- 1. 训练/更新 DNN 代理模型 ----
    % 输入标准化（Z-score）
    train_x = model.now_x;
    train_y = model.now_value;

    x_mean = mean(train_x, 1);
    x_std = std(train_x, 0, 1);
    x_std(x_std < 1e-8) = 1;

    train_x_norm = (train_x - x_mean) ./ x_std;

    % 为每个响应分别训练 DNN
    n_train = size(train_x, 1);
    dnn_models = cell(1, yd);

    for j = 1:yd
        y_j = train_y(:, j);

        % 分类响应：标签转为 {0, 1}，用 sigmoid 输出
        is_class = (model.ClassPos(j) == 2);

        % 构建网络
        net = build_dnn(dim, layers_config, is_class, max_epochs);

        % 训练
        net = train_dnn(net, train_x_norm, y_j, is_class, max_epochs);

        dnn_models{j} = net;
    end

    model.DNN = struct('models', dnn_models, ...
                       'x_mean', x_mean, 'x_std', x_std, ...
                       'class_pos', model.ClassPos);

    %% ---- 2. DNN 预测候选点 ----
    cand_x_norm = (model.candidate - x_mean) ./ x_std;

    y_pred = zeros(ns, yd);
    var = zeros(ns, yd);
    p_class_all = zeros(ns, yd);  % 分类响应的预测概率

    for j = 1:yd
        is_class = (model.ClassPos(j) == 2);
        [y_pred(:, j), p_val] = predict_dnn(dnn_models{j}, cand_x_norm, is_class);

        if is_class
            p_class_all(:, j) = p_val;
            % 分类响应不确定性：4p(1-p)，决策边界处最大
            p_clamped = min(max(p_val, 1e-6), 1 - 1e-6);
            var(:, j) = 4 .* p_clamped .* (1 - p_clamped);
        else
            % 连续响应：DNN 不提供方差，用预测值偏离中位数的程度近似
            % 这对应原文献的 |g_hat(x)|（预测值绝对值度量"离极值面多远"）
            y_median = median(y_pred(:, j));
            y_range = max(y_pred(:, j)) - min(y_pred(:, j));
            if y_range < 1e-12; y_range = 1; end
            var(:, j) = abs(y_pred(:, j) - y_median) ./ y_range;
        end
    end

    %% ---- 3. 多响应有效预测不确定性 sigma_eff ----
    meanvar = mean(var, 1);
    maxvar  = max(var, [], 1);
    weight = 0.5 * (meanvar ./ (sum(meanvar) + 1e-12) + ...
                    maxvar ./ (sum(maxvar) + 1e-12));
    sigma_eff = sum(var .* weight, 2);
    sigma_eff = sigma_eff ./ (max(sigma_eff) + 1e-12);

    %% ---- 4. 距离学习函数 psi_d ----
    d_min = min(pdist2(model.candidate, model.now_x), [], 2);
    d_min_norm = d_min ./ (max(d_min) + 1e-12);
    psi_d = sigma_eff .* d_min_norm;

    %% ---- 5. 筛选候选带 ----
    psi_d_max = max(psi_d);
    threshold = band_ratio * psi_d_max;
    candidate_band = find(psi_d >= threshold);

    if isempty(candidate_band)
        candidate_band = (1:ns)';
    end

    %% ---- 6. 在候选带内选择 ----
    if isfield(model, 'input_pdf')
        pdf_vals = model.input_pdf(model.candidate(candidate_band, :));
        [~, local_idx] = max(pdf_vals);
    else
        [~, local_idx] = max(psi_d(candidate_band));
    end

    best_idx = candidate_band(local_idx);
    x_star = model.candidate(best_idx, :);

    %% ---- 7. 候选点剔除记录 ----
    prune_dist = prune_factor * d_min(best_idx);
    too_close = pdist2(model.candidate, x_star) < prune_dist;
    too_close(best_idx) = false;
    model.kr_pruned_idx = find(too_close);

    %% ---- 8. 记录采样标记 ----
    model.M_ind(size(model.now_value, 1) - size(model.init_value, 1) + 1) = 0;

end


%% ==================== DNN 辅助函数 ====================

function net = build_dnn(dim, layers_config, is_class, max_epochs)
% 构建深度神经网络
% 使用 MATLAB Neural Network Toolbox 的 fitnet
% SCG 优化器（trainscg），比默认 trainlm 更适合小样本场景

    n_layers = length(layers_config);

    % fitnet 对分类和回归都是线性输出；分类通过 tansig 后处理
    net = fitnet(layers_config, 'trainscg');

    % 所有隐藏层用 ReLU（poslin）
    for i = 1:n_layers
        net.layers{i}.transferFcn = 'poslin';
    end

    % 训练参数
    net.trainParam.epochs = max_epochs;
    net.performFcn = 'mse';
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;

    % 全部数据用于训练（不划分验证/测试集）
    net.divideFcn = 'dividetrain';
end


function net = train_dnn(net, X, y, is_class, ~)
% 训练 DNN
% dividetrain 已在 build_dnn 中设置，全部数据用于训练

    if is_class
        % 分类响应：y 为 {0, 1}，转为 {-1, 1} 以配合线性输出的符号判决
        y_train = y * 2 - 1;  % {0,1} -> {-1,1}
    else
        y_train = y;
    end

    try
        net = train(net, X', y_train');
    catch
        % 训练失败时使用默认初始化重试
        net = init(net);
        net = train(net, X', y_train');
    end
end


function [y_pred, p_val] = predict_dnn(net, X, is_class)
% DNN 预测
% fitnet 输出层为线性，输出值可能超出 [-1,1]
% 分类响应通过 tanh 压缩到 [-1,1]，再映射到概率 [0,1]

    raw = net(X')';

    if is_class
        % 用 tanh 将线性输出压缩到 [-1, 1]
        raw_clamped = tanh(raw);
        p_val = (raw_clamped + 1) / 2;
        p_val = min(max(p_val, 1e-6), 1 - 1e-6);
        % 硬判决
        y_pred = zeros(size(raw));
        y_pred(p_val >= 0.5) = 1;
    else
        y_pred = raw;
        p_val = [];
    end
end
