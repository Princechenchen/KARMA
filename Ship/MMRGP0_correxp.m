% MMRGP0_correxp —— MMRGP0 的 correxp 核变体(2026-08-20): 与 numeric_sample/MMRGP0.m 机械派生(sed),
% 唯一差异 = 连续响应训练调 MRSM_correxp。Ackley 粗糙面约定 correxp。原 MMRGP0 零改动。
function [mx] = MMRGP0_correxp(trainy,trainx,ClassPos,dm)
[parent_dir, ~, ~] = fileparts(pwd);
disp ('executing gpml startup script...')
mydir='F:\学习工作\搞点研究\混合响应建模\单一响应建模\';        % where am I located
% mydir='F:\学习工作-wzc\搞点研究\混合响应建模\单一响应建模'; 
addpath (mydir);
% core folders
dirs = {'cov','doc','inf','lik','mean','prior','util'};
for d = dirs
  addpath (fullfile (mydir, d{1}))
end
% minfunc folders
dirs = {{'util','minfunc'},{'util','minfunc','compiled'}};
for d = dirs
  addpath (fullfile (mydir, d{1}{:}))
end
addpath([mydir,'/util/sparseinv'])

trainym = trainy; 
%分类潜变量求解
for d = find(ClassPos==2)
meanfunc = @meanConst; hyp.mean = 0;
covfunc = @covSEard; ell = 1.0; sf = 1.0; hyp.cov = zeros(1,dm+1);  %covLINard 用于max和MSE
likfunc = @likErf; 
hyp = struct('mean', 0, 'cov', zeros(1,dm+1));
zeroindex = find(trainy(:,d)~=1);  %分类响应0替换为-1 %feval(covfunc)
trainy(zeroindex,d)=-1;
hyp = minimize(hyp, @gp, -40, @infLaplace, meanfunc, covfunc, likfunc, trainx, trainy(:,d));             
[ymu,~,fm,~,lp2] = gp(hyp, @infLaplace, meanfunc, covfunc, likfunc, trainx, trainy(:,d), trainx,ones(size(trainx,1), 1));
trainym(:,d)=fm;
end
%混合响应训练
% conmodeldir = '.\MMRGP_subprograms';    %参数变为A   %%！！这个方法模型构建有问题，先别用！！
% conmodeldir = 'F:\学习工作-wzc\搞点研究\混合响应建模\0625混合响应稀疏近似+相关性+多分类尝试\MRSM_subprograms';
conmodeldir = [parent_dir,'\MRSM_subprograms1'];   %优化版（P0系列数值优化，原 MRSM_subprograms0）
addpath(conmodeldir);
[trainym,settings]=mapminmax(trainym',-1,1);
trainym = trainym';
mx = MRSM_correxp(trainx,trainym);%变体唯一改动(原MRSM,correxp核)
mx.settings = settings;
mx.classpos=ClassPos;
% %预测
% predictY = predict_resp(mx,trainx);
% predictY = mapminmax('reverse', predictY', settings); predictY = predictY';
% fc = predictY(:,d); 
% lZ = logphi(fc); p = exp(lZ);  
% ymu = 2*p-1;          %分类响应预测均值   %以0为界，+1>0, -1<0
% ymu(find(ymu>=0)) = 1;
% ymu(find(ymu<0)) = 0;
% predictY(:,d)=ymu;
% rmpath(conmodeldir);
% 预测误差
% error = error_func(trainy,predictY,ClassPos);   %预测误差
% resultx = [error{1},error{2},error{4}];


