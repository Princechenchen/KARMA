function [selected_archive, selected_fitnesses] = select_top_n_pareto(archive, archive_fitnesses, n)
    fronts = non_dominated_sorting(archive_fitnesses);
    selected_indices = [];
    
    for i = 1:length(fronts)
        front = fronts{i};
        if length(selected_indices) + length(front) > n
            crowding_distances = compute_crowding_distance(archive_fitnesses, front);
            [~, sorted_indices] = sort(crowding_distances, 'descend');
            remaining = n - length(selected_indices);
            selected_indices = [selected_indices, front(sorted_indices(1:remaining))];
            break;
        else
            selected_indices = [selected_indices, front];
        end
    end
    
    selected_archive = archive(selected_indices, :);
    selected_fitnesses = archive_fitnesses(selected_indices);
end
