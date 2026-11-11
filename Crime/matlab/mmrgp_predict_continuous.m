function [mu_z,var_z] = mmrgp_predict_continuous(model,query_x)
% Continuous-only equivalent of predict_resp, without mixed covariance blocks.
assert(ismatrix(query_x) && size(query_x,2)==model.d,'query_x must be N-by-d');
cont = find(model.classpos==1,1,'first');
assert(cont==1,'Crimes continuous response must be first');
model.x_pred = query_x;
model.n0 = size(query_x,1);
model.z_0n0n = define_covmatrix0(model,model.x_pred,model.X);
z0 = model.hyper.A*model.hyper.A';
n = model.n0;

alpha1 = model.ymn-model.fn*model.beta;
zchol = poschol(model.z_mn);
alpha2 = zchol\(zchol'\alpha1);
model.f0 = trend_fun(model,model.x_pred);
y = model.f0*model.beta+model.z_0n0n*alpha2;
y = reshape(y,model.m,n)';

alpha3 = zchol\(zchol'\model.fn);
u = model.f0-model.z_0n0n*alpha3;
alpha4 = model.fn'*alpha3;
alpha4chol = poschol(alpha4);
alpha5 = alpha4chol\(alpha4chol'\u');
alpha6 = zchol\(zchol'\model.z_0n0n');
var_z = zeros(n,1);
for i = 1:n
    idx = model.m*(i-1)+cont;
    var_z(i) = z0(cont,cont)-model.z_0n0n(idx,:)*alpha6(:,idx)+u(idx,:)*alpha5(:,idx);
end
scale = (model.settings.xmax(cont)-model.settings.xmin(cont))/2;
y = mapminmax('reverse',y',model.settings); y = y';
mu_z = y(:,cont);
var_z = var_z*scale^2;
end
