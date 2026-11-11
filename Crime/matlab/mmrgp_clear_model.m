function mmrgp_clear_model(model_id)
% Release one fitted model without closing a shared MATLAB Engine.
assert(~isempty(regexp(model_id,'^[A-Za-z0-9_]+$','once')),'Invalid model_id');
name = ['crimes_mmrgp_',model_id];
evalin('base',sprintf('clear %s',name));
end
