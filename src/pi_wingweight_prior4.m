function [pi_hg, pi_hg2, pri] = pi_wingweight_prior4(X, mode)
    if nargin < 2, mode = 'wq'; end
    if nargin >= 1 && ~isempty(X)
        pi_hg = local_pi(X, local_params(mode));
        pri = [];
        return;
    end
    p = local_params(mode);
    pi_hg  = @(Xq) local_pi(Xq, p);
    pi_hg2 = pi_hg;
    pri = struct('mode', mode, 'ew', p.ew, 'anchors', {p.anchors}, 'pi', pi_hg, ...
                 'n_anchor', numel(p.anchors));
end
function p = local_params(mode)
    ew = 0.08;
    a_w = struct('dims', 10, 'thr', 0.25, 'side', +1, ...
                 'label', 'Wp低带(x<=0.25, 物理Wp<=0.0388; 加性项交互盲区)');
    a_q = struct('dims', 5, 'thr', 0.75, 'side', -1, ...
                 'label', 'q高带(x>=0.75, 物理q>=37.75; 平坦维ARD可辨识性)');
    switch lower(mode)
        case 'wq',  anchors = [a_w, a_q];
        case 'w',   anchors = a_w;
        case 'q',   anchors = a_q;
        otherwise,  error('pi_wingweight_prior4: 未知 mode "%s" (wq/w/q)', mode);
    end
    p = struct('ew', ew, 'anchors', {anchors});
end
function y = local_pi(X, p)
    n = size(X, 1);
    y = 0.01 * ones(n, 1);
    for i = 1:numel(p.anchors)
        a = p.anchors(i);
        d = max(a.side * (X(:, a.dims) - a.thr), 0) / p.ew;
        y = max(y, max(0.01, exp(-0.5 * d.^2)));
    end
end
