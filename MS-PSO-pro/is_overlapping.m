function overlap = is_overlapping(new_region, existing_regions)
    overlap = false;
    for i = 1:size(existing_regions, 1)
        % 计算新区域和现有区域的重叠面积
        overlap_area = 1; % 初始化重叠面积为1（假设所有维度都有重叠）
        new_volume = 1; % 初始化新区域的体积
        exist_volume = 1; % 初始化现有区域的体积
        for d = 1:length(new_region) / 2
            new_start = new_region(2 * d - 1);
            new_end = new_region(2 * d);
            exist_start = existing_regions(i, 2 * d - 1);
            exist_end = existing_regions(i, 2 * d);
            
            % 计算当前维度的重叠区间
            overlap_start = max(new_start, exist_start);
            overlap_end = min(new_end, exist_end);
            
            % 如果没有重叠，直接返回false
            if overlap_start >= overlap_end
                overlap_area = 0;
                break;
            end
            
            % 计算当前维度的重叠长度
            overlap_length = overlap_end - overlap_start;
            overlap_area = overlap_area * overlap_length;
            
            % 计算新区域和现有区域的体积
            new_volume = new_volume * (new_end - new_start);
            exist_volume = exist_volume * (exist_end - exist_start);
        end
        
        % 计算重叠比例
        overlap_ratio = overlap_area / min(new_volume, exist_volume);
        
        % 如果重叠比例超过30%，返回true
        if overlap_ratio > 0.4      %0.6Branin0.4也行 0.4Ackley
            overlap = true;
            return;
        end
    end
end