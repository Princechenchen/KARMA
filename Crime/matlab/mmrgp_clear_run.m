function mmrgp_clear_run(model_id)
% Release both the fitted model and all acquisition state for one run.
assert(~isempty(regexp(model_id,'^[A-Za-z0-9_]+$','once')),'Invalid model_id');
evalin('base',sprintf('clear crimes_mmrgp_%s crimes_state_%s',model_id,model_id));
end
