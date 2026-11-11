function mmrgp_save_checkpoint(model_id,checkpoint_path,fit_budget)
% Atomically prepared by Python: persist fitted model and sampler state.
assert(~isempty(regexp(model_id,'^[A-Za-z0-9_]+$','once')),'Invalid model_id');
name=['crimes_mmrgp_',model_id]; state_name=['crimes_state_',model_id];
assert(evalin('base',sprintf('exist(''%s'',''var'')',name))==1,'Model is not fitted');
fitted_model=evalin('base',name); %#ok<NASGU>
if evalin('base',sprintf('exist(''%s'',''var'')',state_name))==1
    sampler_state=evalin('base',state_name); %#ok<NASGU>
else
    sampler_state=[]; %#ok<NASGU>
end
save(checkpoint_path,'fitted_model','sampler_state','fit_budget','-v7');
end
