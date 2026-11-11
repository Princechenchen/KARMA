function points = random_initialization(dim, num_points, lower_bound, upper_bound)
    points = (upper_bound - lower_bound) .* rand(num_points, dim) + lower_bound;
end
