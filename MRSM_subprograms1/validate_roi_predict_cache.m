function validate_roi_predict_cache(model, cache)
% Reject a cache built for a different fitted MMRGP model.
required_fields = {'X', 'ymn', 'fn', 'beta', 'classpos', 'settings', 'A', 'teta'};
if ~isstruct(cache) || ~isfield(cache, 'source')
    error('ROI:StalePredictCache', ...
        'ROI prediction cache is missing its model binding.');
end

current.X = model.X;
current.ymn = model.ymn;
current.fn = model.fn;
current.beta = model.beta;
current.classpos = model.classpos;
current.settings = model.settings;
current.A = model.hyper.A;
current.teta = model.hyper.teta;
for k = 1:numel(required_fields)
    name = required_fields{k};
    if ~isfield(cache.source, name) || ~isequaln(cache.source.(name), current.(name))
        error('ROI:StalePredictCache', ...
            'ROI prediction cache does not match the current MMRGP model.');
    end
end
end
