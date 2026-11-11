function mmrgp_fit_crimes(train_x,train_y,class_pos,gpml_root,model_id)
% Fit once and retain the model in the MATLAB base workspace.
assert(~isempty(regexp(model_id,'^[A-Za-z0-9_]+$','once')),'Invalid model_id');
model = MMRGP0_crimes(train_y,train_x,class_pos,size(train_x,2),gpml_root);
assignin('base',['crimes_mmrgp_',model_id],model);
end
