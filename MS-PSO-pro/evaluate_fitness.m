function fitness = evaluate_fitness(particle, model,num_samples,dim,model_num,occupied_regions)
 

    region_bounds = [];
    new_region = zeros(1, 2*dim);
    for d = 1:dim
        feature_start = particle(2*d - 1);
        feature_size = particle(2*d);
        feature_end = feature_start + feature_size;
        region_bounds = [region_bounds; feature_start, feature_end];
        new_region(2*d-1) = feature_start;
        new_region(2*d) = feature_end;
    end
    
    % 根据是否存在分类响应动态确定目标数
    has_class = isfield(model, 'ClassPos') && any(model.ClassPos == 2);
    if has_class
        num_fitness = 3;
    else
        num_fitness = 2;
    end
    
    if is_overlapping(new_region, occupied_regions)
        fitness = -1e6 * ones(1, num_fitness);
        return;
    end


    sampled_points = latin_hypercube_sampling(region_bounds, num_samples);
    % Calculate function values at the sampled points
%     y_pred = Y(sampled_points, model,model_num);
    %% Define the objective function to be minimized
    func = @(x) Y(x, model,model_num); 
%     extrema = find_local_minima(func, dim, 40, region_bounds(:,1)', region_bounds(:,2)', 'fminunc', 1e-3);
%     num_extrema = length(extrema);
    sampledY = func(sampled_points);
    
    % Calculate the fitness metrics
    information_surface_measure = calculate_entropy(sampled_points, sampledY, 0.05);
    
    gradient_metric = compute_gradient_metric(sampled_points, sampledY);
%     kl_divergence = calculate_kl_divergence_kde(sampled_points, kde_q_, 0.2);

    % 分类响应不确定性（NME，Normalized Mean Entropy）
    class_uncertainty = [];
    if has_class
        class_cols = find(model.ClassPos == 2);
        n_class = length(class_cols);
        u_sum = 0;
        for k = 1:n_class
            fc = sampledY(:, class_cols(k));
            p = (fc + 1) / 2;  % fc in [-1, 1] -> p in [0, 1]
            p = min(max(p, 1e-12), 1 - 1e-12);
            H = -(p .* log(p) + (1 - p) .* log(1 - p));
            u_sum = u_sum + mean(H) / log(2);
        end
        class_uncertainty = u_sum / n_class;
    end

    % Final fitness value
%     fitness = [information_surface_measure,gradient_metric,kl_divergence, num_extrema];%kl_divergence, num_extrema;
    fitness = [information_surface_measure, gradient_metric, class_uncertainty];%kl_divergence, num_extrema;
end
