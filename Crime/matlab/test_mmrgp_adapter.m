function test_mmrgp_adapter(gpml_root)
% Real MATLAB integration test for the portable Crimes MMRGP adapter.
assert(nargin == 1 && isfolder(gpml_root), 'Valid gpml_root is required');
rng(7);
train_x = [linspace(0.05,0.95,12)', repmat(linspace(0.1,0.9,4),1,3)'];
counts = [2;4;3;8;12;7;18;25;11;30;22;35];
labels = double(mod((1:12)',3) == 0);
model = MMRGP0_crimes([log1p(counts),labels],train_x,[1,2],2,gpml_root);
assert(string(model.corrfunc) == "correxp");

[mu,var_z] = mmrgp_predict_continuous(model,train_x(1:4,:));
[y,s] = predict_resp(model,train_x(1:4,:));
assert(isequal(size(s),[8,2]), 'predict_resp covariance layout changed');
idx = 1:2:8;
direct_var = s(idx,1) * ((model.settings.xmax(1)-model.settings.xmin(1))/2)^2;
assert(isequal(size(mu),[4,1]) && isequal(size(var_z),[4,1]));
assert(all(isfinite(mu)) && all(isfinite(var_z)) && all(var_z >= -1e-10));
assert(max(abs(y(:,1)-mu)) < 1e-10);
assert(max(abs(direct_var-var_z)) < 1e-10);

chunked = zeros(4,1);
for i = 1:2
    [~,chunked(2*i-1:2*i)] = mmrgp_predict_continuous(model,train_x(2*i-1:2*i,:));
end
assert(max(abs(chunked-var_z)) < 1e-10);

model_id = 'unittest';
mmrgp_fit_crimes(train_x,[log1p(counts),labels],[1,2],gpml_root,model_id);
[mu_latest,var_latest] = mmrgp_predict_latest(train_x(1:4,:),model_id);
assert(isequal(size(mu_latest),[4,1]) && isequal(size(var_latest),[4,1]));
mmrgp_clear_model(model_id);
assert(evalin('base','exist(''crimes_mmrgp_unittest'',''var'')') == 0);
disp('MMRGP_ADAPTER_MATLAB_PASS');
end
