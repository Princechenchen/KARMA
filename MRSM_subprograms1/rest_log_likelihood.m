function [lml] = rest_log_likelihood(x, model)

% This function calculates the restricted likelihod at each set of hyperparametrs

n1=model.m*(model.m-1)/2;
n2=model.m*model.d;

k=1;

for i=1: model.m
    for j=i+1:model.m
    z0(i,j)=x(k);
    k=k+1;
    end
end

if model.m==1;
    z0=[];
end

z0=[z0;zeros(1,model.m)];
z0=z0+z0'+diag(ones(model.m,1));

[V, D]=eig(z0);
D(D<0)=0;
model.hyper.A=V*(D^0.5)*V';
teta=x(n1+1:end);
teta=reshape(teta,model.m,model.d);
model.hyper.teta=teta;

y=model.ymn;
f=model.fn;

model.z_mn=define_covmatrix0(model,model.X,model.X); % update the value of model. zmn
zmnchol = poschol(model.z_mn);   % 提前计算Cholesky，后面复用
model.beta=find_beta(model, zmnchol); % 传入已分解的zchol，避免find_beta内部重复分解

%% Determine the restricted likelihood function
alpha1=y-f*model.beta;
alpha2 = (zmnchol\(zmnchol'\alpha1));
sigma2=alpha1'*alpha2;
alpha3=(zmnchol\(zmnchol'\f));
alpha4=f'*alpha3;

% RMLE
lml=0.5*(sigma2)+0.5*log(det(model.z_mn))+0.5*log(det(alpha4));
if isinf(lml)
lml=0.5*(sigma2)+log(sum(diag(zmnchol)))+0.5*log(det(alpha4));
% MLE
% lml=0.5*(sigma2)+0.5*log(det(model.z_mn));
end