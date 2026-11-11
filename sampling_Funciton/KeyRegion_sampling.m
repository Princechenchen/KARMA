function [x_star, model] = KeyRegion_sampling(model)
% KeyRegion_sampling  数据驱动重点区域采样（C9-MMRGP版本）
%
% 文献来源：
%   李宁, 张孝航. 基于重点区域采样深度神经网络代理模型的可靠性分析方法.
%   工程力学, 2026, 43(5): 24-32.
%
% 原始方法使用DNN代理模型，面向单标量极限状态函数。
% 本版本将其适配到MMRGP混合响应（连续+分类）场景：
%   - 代理模型替换为MMRGP（与项目所有方法共享，保证公平比较）
%   - 距离学习函数 |g_hat(x)| 替换为多响应有效预测不确定性 sigma_eff(x)
%   - 分类响应不确定性用 4p(1-p)（决策边界处最大）
%
% 采样决策逻辑（与原文献一致）：
%   1. 距离学习函数 psi_d = sigma_eff / d_min（值大 = 不确定性高 + 远离已有样本）
%   2. 筛选 psi_d 最高的候选带（>= band_ratio * max(psi_d)）
%   3. 在候选带内按输入概率密度 f(x) 选择（均匀输入下取 psi_d 最大者）
%   4. 候选点剔除规则：防止采样聚集
%
% 输入：
%   model.candidate  - ns x dim  候选点集
%   model.now_x      - n  x dim  已有训练样本
%   model.now_value  - n  x m    已有训练响应
%   model.ClassPos   - 1 x m    响应类型（1=连续, 2=分类）
%   model.MMRGP      -          MMRGP代理模型
%
% 输出：
%   x_star           - 1 x dim  选定的下一个采样点
%   model            -          更新后的模型（含 M_ind 标记）

    ns = size(model.candidate, 1);
    yd = size(model.init_value, 2);   % 响应维度 m

    %% ---- 参数设置（可在 model 中覆盖）----
    % 候选带筛选比例：保留 psi_d >= band_ratio * max(psi_d) 的候选点
    % 原文献取 2*psi_min 阈值（约保留前 30-50%），此处用比例等价
    if isfield(model, 'kr_band_ratio')
        band_ratio = model.kr_band_ratio;
    else
        band_ratio = 0.5;
    end

    % 候选点剔除阈值：新点响应接近失效面时，剔除距离过近的候选点
    % delta_threshold 对应原文献的 Delta（0.01~0.1）
    if isfield(model, 'kr_delta')
        delta_threshold = model.kr_delta;
    else
        delta_threshold = 0.05;
    end

    % 剔除距离因子：距离新点 < prune_factor * d_min 的候选被剔除
    if isfield(model, 'kr_prune_factor')
        prune_factor = model.kr_prune_factor;
    else
        prune_factor = 0.5;
    end

    %% ---- 1. MMRGP 预测 ----
    [y_pred, s_pred, model.MMRGP] = predict_resp(model.MMRGP, model.candidate);

    %% ---- 2. 计算多响应有效预测不确定性 sigma_eff ----
    % 从协方差矩阵 s 中提取各响应对角元素（预测方差）
    var = zeros(ns, yd);
    for i = 1:yd
        var(:, i) = s_pred(i:yd:end, i);
    end

    % 分类响应：用预测概率的方差 4p(1-p) 替代预测方差
    % 决策边界处 p≈0.5，方差最大；远离边界处方差趋零
    class_cols = find(model.ClassPos == 2);
    for k = 1:length(class_cols)
        col = class_cols(k);
        % predict_resp 设置 model.MMRGP.fmu 为连续潜变量预测值（-1 到 +1），
        % 注意：model.MMRGP.fc 是硬判决后的 0/1 值，不能用于计算不确定性。
        % fmu 的列按分类响应顺序排列（仅含 ClassPos==2 的列）。
        if isfield(model.MMRGP, 'fmu') && ~isempty(model.MMRGP.fmu)
            fmu_pred = model.MMRGP.fmu;   % ns x n_class
            p_class = (fmu_pred(:, k) + 1) / 2;  % 潜变量 -> 概率
            p_class = min(max(p_class, 1e-6), 1 - 1e-6);
            var(:, col) = 4 .* p_class .* (1 - p_class);
        end
    end

    % 多响应归一化 + 自适应加权（均值-最大值方案，与 mROI_SUR2w 一致）
    meanvar = mean(var, 1);   % 1 x m
    maxvar  = max(var, [], 1); % 1 x m
    weight = 0.5 * (meanvar ./ sum(meanvar) + maxvar ./ sum(maxvar));
    sigma_eff = sum(var .* weight, 2);  % ns x 1

    % 归一化到 [0, 1]
    sigma_eff = sigma_eff ./ (max(sigma_eff) + 1e-12);

    %% ---- 3. 计算距离学习函数 psi_d ----
    % 原文献: psi_d = |g_hat(x)| / d_min(x)，值越小越值得采样（靠近失效面 + 远离样本）
    % 适配后: 改为不确定性驱动，psi_d = sigma_eff * d_min_norm，值越大越值得采样
    % （不确定性高 + 距离已有样本远 = 越需要补数据）
    d_min = min(pdist2(model.candidate, model.now_x), [], 2);  % ns x 1
    d_min_norm = d_min ./ (max(d_min) + 1e-12);

    psi_d = sigma_eff .* d_min_norm;   % 两者都是 ns×1 列向量，元素级乘法

    %% ---- 4. 筛选候选带 ----
    psi_d_max = max(psi_d);
    threshold = band_ratio * psi_d_max;
    candidate_band = find(psi_d >= threshold);

    if isempty(candidate_band)
        candidate_band = (1:ns)';
    end

    %% ---- 5. 在候选带内按概率密度选择 ----
    % 输入空间为 [0,1]^dim 的均匀分布时，f(x) 恒定，退化为取 psi_d 最大
    % 若模型提供了非均匀概率密度函数，则按密度选择
    if isfield(model, 'input_pdf')
        % 使用用户提供的概率密度函数句柄
        pdf_vals = model.input_pdf(model.candidate(candidate_band, :));
        [~, local_idx] = max(pdf_vals);
    else
        % 均匀分布：取候选带内 psi_d 最大的点
        [~, local_idx] = max(psi_d(candidate_band));
    end

    best_idx = candidate_band(local_idx);
    x_star = model.candidate(best_idx, :);

    %% ---- 6. 候选点剔除规则（供下一轮使用，记录到 model）----
    % 原文献：若新采样点真实响应接近失效面，剔除距离过近的候选点
    % 此处在当前轮次记录剔除信息，不修改候选池（外层循环统一管理）
    prune_dist = prune_factor * d_min(best_idx);
    too_close = pdist2(model.candidate, x_star) < prune_dist;
    too_close(best_idx) = false;  % 不剔除选中的点本身
    model.kr_pruned_idx = find(too_close);  % 记录被剔除的候选索引

    %% ---- 7. 记录采样标记 ----
    model.M_ind(size(model.now_value, 1) - size(model.init_value, 1) + 1) = 0;

end
