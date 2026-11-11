function unique_local_minima = find_local_minima(func, dim, num_points, lower_bound, upper_bound, method, tolerance)
    if nargin < 6
        method = 'fminunc';  % Default method
    end
    if nargin < 7
        tolerance = 1e-5;  % Default tolerance
    end
    initial_points = random_initialization(dim, num_points, lower_bound, upper_bound);
    local_minima = local_optimization(func, initial_points, method);
    unique_local_minima = remove_duplicates(local_minima, tolerance);
end