function val = hyperrect_prior(X, centers, hw, ew)
eps_floor = 0.01;
X = reshape(X, [], size(centers,2));
val = zeros(size(X,1),1);
for j = 1:size(centers,1)
    dx = max(abs(X - centers(j,:)) - hw, 0);
    d  = sqrt(sum(dx.^2, 2));
    sc = exp(-0.5*(d/ew).^2);
    val = max(val, sc);
end
val = min(max(val, eps_floor), 1);
end
