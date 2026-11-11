function [pi_a_fn, pi_a_fn2, pri] = pi_wingweight_prior3(X)
if nargin == 1
    pi_a_fn = local_pi_a(X, local_params());
    pri = [];
    return;
end
p = local_params();
pi_a_fn  = @(Xq) local_pi_a(Xq, p);
pi_a_fn2 = pi_a_fn;
pri = struct('ew', p.ew, 'anchors', {p.anchors}, 'pi', pi_a_fn, 'n_anchor', 1);
end
function p = local_params()
anchors = struct('dims', {3}, 'thr', {0.25}, 'side', {+1}, ...
                 'label', {'A(小展弦比带 A<=0.25, 物理A<=7.0)'});
p = struct('ew', 0.08, 'anchors', {anchors});
end
function y = local_pi_a(X, p)
a = p.anchors(1);
d = max(a.side * (X(:, a.dims) - a.thr), 0) / p.ew;
y = max(0.01, exp(-0.5 * d.^2));
end
