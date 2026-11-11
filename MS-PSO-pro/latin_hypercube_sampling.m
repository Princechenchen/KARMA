function samples = latin_hypercube_sampling(bounds, n, unit_samples)
% Generate a randomized Latin hypercube and map it to the requested bounds.
[dim, ~] = size(bounds);
if nargin < 3 || isempty(unit_samples)
    unit_samples = lhsdesign(n, dim, 'criterion', 'none');
else
    if ~isequal(size(unit_samples), [n, dim])
        error('ROI:InvalidSamplingTemplate', ...
            'The unit ROI sampling template must have size n-by-dim.');
    end
end
lower_bounds = bounds(:, 1)';
samples = lower_bounds + unit_samples .* (bounds(:, 2) - bounds(:, 1))';
end
