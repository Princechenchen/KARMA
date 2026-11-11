function overlap = is_overlapping_batch(new_regions, existing_regions)
% Vectorized equivalent of is_overlapping for multiple candidate regions.
num_new = size(new_regions, 1);
overlap = false(num_new, 1);
if num_new == 0 || isempty(existing_regions)
    return;
end

dim = size(new_regions, 2) / 2;
new_start = new_regions(:, 1:2:end);
new_end = new_regions(:, 2:2:end);
existing_start = existing_regions(:, 1:2:end);
existing_end = existing_regions(:, 2:2:end);

new_volume = prod(new_end - new_start, 2);
existing_volume = prod(existing_end - existing_start, 2)';
overlap_start = max(reshape(new_start, num_new, 1, dim), ...
    reshape(existing_start, 1, size(existing_regions, 1), dim));
overlap_end = min(reshape(new_end, num_new, 1, dim), ...
    reshape(existing_end, 1, size(existing_regions, 1), dim));
overlap_lengths = max(0, overlap_end - overlap_start);
overlap_volume = prod(overlap_lengths, 3);
overlap_ratio = overlap_volume ./ min(new_volume, existing_volume);
overlap(:) = any(overlap_ratio > 0.4, 2);
end
