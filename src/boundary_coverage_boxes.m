function boxes = boundary_coverage_boxes(g_fn, thr, bounds, nb, frac, grid_n, pad)
    if nargin < 4 || isempty(nb),     nb     = 3;       end
    if nargin < 5 || isempty(frac),   frac   = [0 0.55]; end
    if nargin < 6 || isempty(grid_n), grid_n = 240;     end
    if nargin < 7 || isempty(pad),    pad    = 0.05;    end
    xv = linspace(bounds(1,1), bounds(1,2), grid_n);
    yv = linspace(bounds(2,1), bounds(2,2), grid_n);
    [Xg, Yg] = meshgrid(xv, yv);
    G = reshape(g_fn([Xg(:), Yg(:)]), grid_n, grid_n);
    C = contourc(xv, yv, G, [thr thr]);
    pts = [];
    k = 1;
    while k <= size(C,2)
        npts = C(2,k);
        pts = [pts; C(1,k+1:k+npts).', C(2,k+1:k+npts).'];
        k = k + npts + 1;
    end
    if size(pts,1) < 2*nb
        error('boundary_coverage_boxes: 等值线点太少(n=%d),请增大 grid_n', size(pts,1));
    end
    arc = [0; cumsum(sqrt(sum(diff(pts,1,1).^2, 2)))];
    t = arc ./ arc(end);
    keep = t >= frac(1) & t <= frac(2);
    pk = pts(keep,:);
    tk = t(keep);
    if size(pk,1) < 2*nb
        error('boundary_coverage_boxes: frac 段内点太少(n=%d),放宽 frac 或增大 grid_n', size(pk,1));
    end
    boxes = zeros(nb,4);
    edges = linspace(frac(1), frac(2), nb+1);
    for b = 1:nb
        if b < nb
            sel = tk >= edges(b) & tk < edges(b+1);
        else
            sel = tk >= edges(b) & tk <= edges(b+1);
        end
        q = pk(sel,:);
        boxes(b,:) = [min(q(:,1))-pad, max(q(:,1))+pad, min(q(:,2))-pad, max(q(:,2))+pad];
    end
    hi = [bounds(1,2), bounds(1,2), bounds(2,2), bounds(2,2)];
    lo = [bounds(1,1), bounds(1,1), bounds(2,1), bounds(2,1)];
    boxes = min(max(boxes, lo), hi);
end
