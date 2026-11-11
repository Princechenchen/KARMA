function fit_budget=mmrgp_load_checkpoint(model_id,checkpoint_path)
% Restore one fitted model and its acquisition state into the base workspace.
assert(~isempty(regexp(model_id,'^[A-Za-z0-9_]+$','once')),'Invalid model_id');
loaded=load(checkpoint_path,'fitted_model','sampler_state','fit_budget');
assert(isscalar(loaded.fit_budget) && loaded.fit_budget>=1,'Invalid checkpoint budget');
name=['crimes_mmrgp_',model_id]; state_name=['crimes_state_',model_id];
assignin('base',name,loaded.fitted_model);
if ~isempty(loaded.sampler_state)
    assignin('base',state_name,loaded.sampler_state);
else
    evalin('base',sprintf('clear %s',state_name));
end
fit_budget=double(loaded.fit_budget);
end
