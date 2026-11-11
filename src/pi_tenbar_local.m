function [pi_local, pri] = pi_tenbar_local(X)
    if nargin < 1
        pi_local = @(Xq) get_local(Xq);
        pri = local_params();
        return
    end
    pi_local = get_local(X);
    pri = local_params();
end
function p = local_params()
    p = struct('dims', [1 3], 'thr', 0.08, 'gamma', 0.30, ...
               'note', 'L2-only boost for y1(rod1)/y2(rod3) extreme thinness');
end
function y = get_local(X)
    p = local_params();
    n = size(X, 1);
    y = zeros(n, 1);
    for a = 1:numel(p.dims)
        xi = max(X(:, p.dims(a)), 0.005);
        y = max(y, min(1, p.thr ./ xi));
    end
end
