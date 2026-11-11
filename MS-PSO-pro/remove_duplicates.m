function unique_minima = remove_duplicates(local_minima, tolerance)
    if nargin < 2
        tolerance = 1e-5;
    end
    unique_minima = [];
    for i = 1:size(local_minima, 1)
        minima = local_minima(i, :);
        is_unique = true;
        for j = 1:size(unique_minima, 1)
            if norm(minima(1:end-1) - unique_minima(j, 1:end-1)) < tolerance
                is_unique = false;
                break;
            end
        end
        if is_unique
            unique_minima = [unique_minima; minima];
        end
    end
end
