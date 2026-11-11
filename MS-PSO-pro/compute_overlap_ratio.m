function ratio = compute_overlap_ratio(region1, region2, dim)
    % 计算两个多维超矩形区域之间的重叠比例
    % 重叠比例 = 重叠体积 / min(区域1体积, 区域2体积)
    %
    % 输入：
    %   region1, region2 - [start1, end1, start2, end2, ..., startD, endD]
    %   dim              - 维度数
    % 输出：
    %   ratio - 重叠比例，范围 [0, 1]

    overlap_volume = 1;
    volume1 = 1;
    volume2 = 1;

    for d = 1:dim
        s1 = region1(2 * d - 1);
        e1 = region1(2 * d);
        s2 = region2(2 * d - 1);
        e2 = region2(2 * d);

        overlap_start = max(s1, s2);
        overlap_end = min(e1, e2);

        % 任一维度无重叠，则整体无重叠
        if overlap_end <= overlap_start
            ratio = 0;
            return;
        end

        overlap_volume = overlap_volume * (overlap_end - overlap_start);
        volume1 = volume1 * (e1 - s1);
        volume2 = volume2 * (e2 - s2);
    end

    min_volume = min(volume1, volume2);
    if min_volume <= 0
        ratio = 0;
    else
        ratio = overlap_volume / min_volume;
    end
end
