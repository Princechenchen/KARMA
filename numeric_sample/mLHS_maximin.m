function [x_star, model] = mLHS_maximin(model)
% mLHS_maximin —— 对比方法 C1: Maximin-LHS 无反馈基线
% 机制(见 对比方法与实验设计计划.md §2.1):
%   一次性预生成与剩余预算等长的 LHS 设计(在 R 个随机 LHS 设计中选
%   [初始点 ∪ 设计点] 整体最小距离最大者,即 maximin 增广),此后每步
%   只从队列头弹出一点,不做任何模型反馈 —— 用作"纯设计论"下界参照。
% 接口与其余对比方法统一: [x_star, model] = f(model), 真值评估由驱动器计费。
% rng 说明: 队列仅在首步生成一次,消耗随机数的时机由驱动器的臂内 rng 重置
%   协议控制,与其余臂配对。

n_seq = model.total_iter - model.n_init;

% 已有队列(第 2 步及以后): 弹出队头即可
if isfield(model, 'lhs_queue') && ~isempty(model.lhs_queue)
    x_star = model.lhs_queue(1, :);
    model.lhs_queue(1, :) = [];
    return;
end

% 首步: R 个候选 LHS 设计,选 maximin 增广(相对固定初始设计)
R = 20;                                   % 设计重启动次数
best = []; bestscore = -inf;
for r = 1:R
    cand = lhsdesign(n_seq, model.dim);
    D = pdist2([model.now_x; cand], [model.now_x; cand]);
    D(1:size(D,1)+1:end) = inf;           % 排除自身距离
    s = min(D(:));
    if s > bestscore
        bestscore = s; best = cand;
    end
end
model.lhs_queue = best;
x_star = model.lhs_queue(1, :);
model.lhs_queue(1, :) = [];
end
