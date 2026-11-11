function [x_star, model] = mLHS_maximin_pool(model)
n_seq = model.total_iter - model.n_init;
if isfield(model, 'lhs_pool_queue') && ~isempty(model.lhs_pool_queue)
    x_star = model.lhs_pool_queue(1, :);
    model.lhs_pool_queue(1, :) = [];
    return;
end
pool = model.candidate;
S = model.now_x;
q = zeros(n_seq, size(pool, 2));
used = false(size(pool, 1), 1);
for k = 1:n_seq
    D = pdist2(pool, S);
    D(used, :) = -inf;
    [~, j] = max(min(D, [], 2));
    q(k, :) = pool(j, :);
    used(j) = true;
    S = [S; pool(j, :)];
end
model.lhs_pool_queue = q;
x_star = model.lhs_pool_queue(1, :);
model.lhs_pool_queue(1, :) = [];
end
