function [archive, archive_fitnesses, did_truncate] = update_archive_batch( ...
    archive, archive_fitnesses, new_positions, new_fitnesses, max_size)
% Merge one generation, filter non-dominated points, and truncate once.
if isempty(new_positions)
    did_truncate = false;
    return;
end

all_positions = [archive; new_positions];
all_fitness = [cell2mat(archive_fitnesses); new_fitnesses];
n = size(all_fitness, 1);
dominates_matrix = true(n, n);
strictly_better = false(n, n);
for objective = 1:size(all_fitness, 2)
    values = all_fitness(:, objective);
    dominates_matrix = dominates_matrix & (values >= values');
    strictly_better = strictly_better | (values > values');
end
dominates_matrix = dominates_matrix & strictly_better;
keep = ~any(dominates_matrix, 1)';

archive = all_positions(keep, :);
fitness_matrix = all_fitness(keep, :);
archive_fitnesses = mat2cell(fitness_matrix, ones(size(fitness_matrix, 1), 1), ...
    size(fitness_matrix, 2));

did_truncate = size(archive, 1) > max_size;
if did_truncate
    crowding = compute_crowding_distance(archive_fitnesses);
    [~, order] = sort(crowding, 'descend');
    keep_idx = order(1:max_size);
    archive = archive(keep_idx, :);
    archive_fitnesses = archive_fitnesses(keep_idx);
end
end
