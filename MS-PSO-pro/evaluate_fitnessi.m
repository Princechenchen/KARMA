function fitness = evaluate_fitnessi(particle, model,num_samples,dim,model_num,occupied_regions)
 

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
    
    if is_overlapping(new_region, occupied_regions)
        fitness = [-1e6, -1e6];
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
    information_surface_measure = calculate_entropy(sampled_points, sampledY(:,model.r), 0.05);
    
    gradient_metric = compute_gradient_metric(sampled_points, sampledY(:,model.r));
%     kl_divergence = calculate_kl_divergence_kde(sampled_points, kde_q_, 0.2);
    % Final fitness value
%     fitness = [information_surface_measure,gradient_metric,kl_divergence, num_extrema];%kl_divergence, num_extrema;
    fitness = [information_surface_measure,gradient_metric];%kl_divergence, num_extrema;
end
