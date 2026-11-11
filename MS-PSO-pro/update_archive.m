function [archive, archive_fitnesses, did_truncate] = update_archive(archive, archive_fitnesses, new_pos, new_fit, max_size)
    did_truncate = false;
    % 非支配解判断
    dominated = false;
    remove_idx = [];
    for i = 1:length(archive_fitnesses)
        if dominates(archive_fitnesses{i}, new_fit)
            dominated = true;
            break;
        elseif dominates(new_fit, archive_fitnesses{i})
            remove_idx = [remove_idx, i];
        end
    end
    if ~dominated
        archive = [archive; new_pos];
        archive_fitnesses = [archive_fitnesses; {new_fit}];
        archive(remove_idx,:) = [];
        archive_fitnesses(remove_idx) = [];
    end
    
    % 拥挤度控制
    if size(archive,1) > max_size
        did_truncate = true;
        fronts = non_dominated_sorting(archive_fitnesses);
        first_front = fronts{1};
        crowding = compute_crowding_distance(archive_fitnesses(first_front));
        [~, idx] = sort(crowding, 'descend');
        keep_idx = idx(1:max_size);
        archive = archive(first_front(keep_idx),:);
        archive_fitnesses = archive_fitnesses(first_front(keep_idx));
    end
end
