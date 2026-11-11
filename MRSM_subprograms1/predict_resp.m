function [y, s, model,s0]=predict_resp(model,x)

model.x_pred=x;
model.n0=size(x,1);
model.z_0n0n=define_covmatrix0(model,model.x_pred,model.X);
z0=model.hyper.A*model.hyper.A';
n=size(model.x_pred,1);
model.z0=repmat(z0,n,1);

%% Prediction at new points
alpha1=model.ymn-model.fn*model.beta;
zchol=poschol(model.z_mn);
alpha2 = (zchol\(zchol'\alpha1));
model.f0=trend_fun(model,model.x_pred);
y=model.f0*model.beta+model.z_0n0n*alpha2;
y=reshape(y,model.m,model.n0)';

%% uncertainity in preictoin at new points
alpha3=(zchol\(zchol'\model.fn));
u=model.f0-model.z_0n0n*alpha3;
alpha4=model.fn'*alpha3;
alpha4chol=poschol(alpha4);
alpha5=(alpha4chol\(alpha4chol'\u'));
alpha6=(zchol\(zchol'\model.z_0n0n'));

for i=1:n
    for j=1:model.m
        for jj=1:model.m   
            alpha7(model.m*(i-1)+1+(j-1),jj)=model.z_0n0n(model.m*(i-1)+1+(j-1),:)*alpha6(:,model.m*(i-1)+jj);
            alpha8(model.m*(i-1)+1+(j-1),jj)=u(model.m*(i-1)+1+(j-1),:)*alpha5(:,model.m*(i-1)+jj);
        end
    end
end
s=model.z0-alpha7+alpha8;
s0 = model.z0-alpha7;
%% Prediciton classification response
fmu = y(:,find(model.classpos==2)); %[fmu]=mapminmax(fmu',-1,1);fmu = fmu'; 
% fmu = (fmu - mean(model.Y(:,model.classpos==2))) ./ std(model.Y(:,model.classpos==2));
y = mapminmax('reverse', y', model.settings); y = y';
% %预测分类响应
fs2 = s(find(model.classpos==2),find(model.classpos==2));
% fs2 = sigma(find(model.classpos==2),find(model.classpos==2));
z = fmu./sqrt(1+fs2);
lZ = logphi(z);       %带入压缩函数
% z = fmu;
% lZ = -log(1+exp(-1*z));
p = exp(lZ);          %预测概率  以0.5为界
ymu = 2*p-1;          %分类响应预测均值   %以0为界，+1>0, -1<0
ys2 = 4*p.*(1-p);     %预测方差
% 
% % ymu(ymu<0.5) = -1;
% % ymu(ymu>=0.5) = 1;
model.fmu = fmu;
model.fc = ymu;
ymu(find(ymu>=0)) = 1;
ymu(find(ymu<0)) = 0;
y(:,find(model.classpos==2)) = ymu;
end