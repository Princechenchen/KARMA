function [mu_z,var_z] = mmrgp_predict_latest(query_x,model_id)
% Predict continuous log1p-response moments from a fitted adapter model.
assert(~isempty(regexp(model_id,'^[A-Za-z0-9_]+$','once')),'Invalid model_id');
name = ['crimes_mmrgp_',model_id];
assert(evalin('base',sprintf('exist(''%s'',''var'')',name))==1,'Model is not fitted');
model = evalin('base',name);
[mu_z,var_z] = mmrgp_predict_continuous(model,query_x);
end
