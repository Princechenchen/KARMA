function cache = prepare_roi_predict_cache(model)
% Cache model-only solves reused by every ROI fitness evaluation.
cache.class_cols = find(model.classpos == 2);
if numel(cache.class_cols) ~= 1
    error('ROI:UnsupportedClassificationCount', ...
        'The ROI Fast path requires exactly one classification response.');
end
cache.zchol = poschol(model.z_mn);
cache.alpha2 = cache.zchol \ ...
    (cache.zchol' \ (model.ymn - model.fn * model.beta));
cache.alpha3 = cache.zchol \ (cache.zchol' \ model.fn);
cache.alpha4chol = poschol(model.fn' * cache.alpha3);
cache.settings = model.settings;
cache.source.X = model.X;
cache.source.ymn = model.ymn;
cache.source.fn = model.fn;
cache.source.beta = model.beta;
cache.source.classpos = model.classpos;
cache.source.settings = model.settings;
cache.source.A = model.hyper.A;
cache.source.teta = model.hyper.teta;
end
