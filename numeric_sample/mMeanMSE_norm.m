function [x_star, model] = mMeanMSE_norm(model)
% mMeanMSE_norm —— 对比方法 C2: N-MeanMSE(归一化均值方差采样)
% 相对原始 sampling_Funciton/meanMSE.m 的唯一改动(计划 §2.2 P0 修复):
%   跨响应归一化 —— 各响应的预测方差先除以该响应在候选池内的最大方差,
%   再做平均,避免量纲差异导致某一响应独占采样决策。归一化规则显式声明:
%   scale_j = max_{x∈候选池} Var_j(x), metric = mean_j Var_j(x)/scale_j。
% 其余保持原方法语义: 全局候选池、逐点 argmax、无 ROI、无先验、无距离项。

ns = size(model.candidate, 1);
yd = size(model.init_value, 2);

[~, s, model.MMRGP] = predict_resp(model.MMRGP, model.candidate);

var = zeros(ns, yd);
for i = 1:yd
    var(:, i) = s(i:yd:end, i);          % 第 i 个响应在全部候选点的预测方差
end
for i = 1:yd
    mx = max(var(:, i));
    if mx <= eps, mx = 1; end
    var(:, i) = var(:, i) / mx;
end

metric = mean(var, 2);
[~, index] = max(metric);
x_star = model.candidate(index, :);

% 与项目其余采样器一致: 记录选中点在候选池中的下标轨迹
model.M_ind(size(model.now_value, 1) - size(model.init_value, 1) + 1) = index;
end
