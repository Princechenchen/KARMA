function [x_star, model] = mLHS_maximin_pool(model)
% mLHS_maximin_pool —— C1 对比方法的"固定数据集"变体（Radar 案例用，原 mLHS_maximin 不动）
% 机制与 mLHS_maximin 同语义（无模型反馈的空间填充下界参照），唯一区别：
%   原版在连续空间 lhsdesign 生成设计点 —— 固定数据集案例没有真值函数可评估，
%   改为在候选池 model.candidate0 上做【贪心 maximin 增广排序】：
%   从 init 设计出发，每步把"离已选集最远"的池点加入队列，直至排完整个池。
%   此后每步弹队头一点。确定性贪心，不消耗 rng（配对协议不依赖其随机性）。
% 接口与其余对比方法统一: [x_star, model] = f(model)，真值查表由驱动器完成。

n_seq = model.total_iter - model.n_init;

% 已有队列(第 2 步及以后): 弹出队头即可
if isfield(model, 'pool_queue') && ~isempty(model.pool_queue)
    x_star = model.pool_queue(1, :);
    model.pool_queue(1, :) = [];
    return;
end

% 首步: 候选池贪心 maximin 增广排序(相对初始设计)
pool = model.candidate0;
if size(pool,1) < n_seq
    error('mLHS_maximin_pool: 候选池(%d)小于序贯步数(%d)', size(pool,1), n_seq);
end
selected = model.now_x;                       % 已选集 = 初始设计
q = zeros(n_seq, size(pool,2));
rest = pool;
for k = 1:n_seq
    D = pdist2(rest, selected);               % 剩余池点 × 已选集
    dmin = min(D, [], 2);                     % 每点到已选集的最小距离
    [~, idx] = max(dmin);                     % 取最远的
    q(k,:) = rest(idx,:);
    selected = [selected; rest(idx,:)];       %#ok<AGROW>
    rest(idx,:) = [];
end
model.pool_queue = q;
x_star = model.pool_queue(1, :);
model.pool_queue(1, :) = [];
end
