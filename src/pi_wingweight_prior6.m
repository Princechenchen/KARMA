function [pi_hg, pi_hg2, pri] = pi_wingweight_prior6(X)
    P.BOX_LO = [0.042, 0.359, 0.650, 0.150, 0.419, 0.324, 0.238, 0.224, 0.563, 0.407];
    P.BOX_HI = [0.350, 0.614, 0.865, 0.827, 0.532, 0.849, 0.350, 0.350, 0.777, 0.754];
    if nargin >= 1 && ~isempty(X)
        pi_hg = local_pi(X, P);
        pri = [];
        return;
    end
    pi_hg  = @(Xq) local_pi(Xq, P);
    pi_hg2 = pi_hg;
    pri = P;
    pri.label = 'HDR针盒(小Sw×高A×高Wdg构型角, vol=7.16e-7)';
end
function y = local_pi(X, P)
    n = size(X, 1);
    inb = true(n, 1);
    for d = 1:10
        inb = inb & (X(:, d) >= P.BOX_LO(d)) & (X(:, d) <= P.BOX_HI(d));
    end
    y = 0.01 * ones(n, 1);
    y(inb) = 1.0;
end
