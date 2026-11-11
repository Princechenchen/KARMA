function fronts = non_dominated_sorting(fitnesses)
    n = length(fitnesses);
    domination_count = zeros(n,1);
    dominated_set = cell(n,1);
    fronts = cell(1,1);
    fronts{1} = [];
    
    for i = 1:n
        for j = i+1:n
            if dominates(fitnesses{i}, fitnesses{j})
                domination_count(j) = domination_count(j) + 1;
                dominated_set{i} = [dominated_set{i}, j];
            elseif dominates(fitnesses{j}, fitnesses{i})
                domination_count(i) = domination_count(i) + 1;
                dominated_set{j} = [dominated_set{j}, i];
            end
        end
        if domination_count(i) == 0
            fronts{1} = [fronts{1}, i];
        end
    end
    
    k = 1;
    while ~isempty(fronts{k})
        next_front = [];
        for i = fronts{k}
            for j = dominated_set{i}
                domination_count(j) = domination_count(j) - 1;
                if domination_count(j) == 0
                    next_front = [next_front, j];
                end
            end
        end
        k = k + 1;
        fronts{k} = next_front;
    end
end
