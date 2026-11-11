function crowding = compute_crowding_distance(fitnesses)
    n = length(fitnesses);
    m = size(fitnesses{1},2);
    matrix = cell2mat(fitnesses);
    crowding = zeros(n,1);
    
    for obj = 1:m
        [~, idx] = sort(matrix(:,obj));

        norm = matrix(idx(end),obj) - matrix(idx(1),obj);
        if norm == 0, continue; end

        % 边界解不再赋予 Inf，改用 2 倍最近邻距离。
        % 这样既能保留边界解的一定优先级，又避免它们总是垄断 top-k 选择，
        % 让位于目标空间稀疏区域的内部解也有机会入选。
        if n >= 2
            crowding(idx(1)) = crowding(idx(1)) + inf+...
                2 * (matrix(idx(2),obj) - matrix(idx(1),obj)) / norm;
            crowding(idx(end)) = crowding(idx(end)) + inf+...
                2 * (matrix(idx(end),obj) - matrix(idx(end-1),obj)) / norm;
        end

        for i = 2:n-1
            crowding(idx(i)) = crowding(idx(i)) + ...
                (matrix(idx(i+1),obj) - matrix(idx(i-1),obj)) / norm;
        end
    end
end