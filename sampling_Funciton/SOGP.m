function [mx] = SOGP(trainy,trainx,ClassPos,dm)

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

% trainym = trainy; 
% [trainym,settings]=mapminmax(trainym',-1,1);
% trainy = trainym';
% mx.settings = settings;
for d=1:size(trainy,2)
if ClassPos(d)~=2
    meanfunc = @meanConst; covfunc = @covSEiso; likfunc = @likGauss;              
    hyp = struct('mean', 0, 'cov', zeros(1,2), 'lik',1);
    hyp = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, trainx, trainy(:,d));
%     [prey s2, ~, ~, lp, post] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, trainx, trainy(:,d), testx);
    mx.meanfunc= meanfunc; mx.inffunc=@infGaussLik; mx.covfunc=covfunc; mx.likfunc =likfunc; mx.hyp = hyp;
else
%分类潜变量求解
% d = find(ClassPos==2);
meanfunc2 = @meanConst; hyp2.mean = 0; 
covfunc2 = @covSEard; ell = 1.0; sf = 1.0; hyp2.cov = zeros(1,dm+1); %covLINard covSEard
likfunc2 = @likErf;  %likGauss likErf
% hyp2 = struct('mean', 0, 'cov', -1*ones(1,dm));
hyp2 = struct('mean', 0, 'cov', zeros(1,dm+1));
zeroindex = find(trainy(:,d)~=1);  %分类响应0替换为-1 %feval(covfunc)
trainy(zeroindex,d)=-1;
hyp2 = minimize(hyp2, @gp, -2, @infLaplace, meanfunc2, covfunc2, likfunc2, trainx, trainy(:,d));             
% [ymu,~,fm,fs2,lp2] = gp(hyp2, @infLaplace, meanfunc2, covfunc2, likfunc2, trainx, trainy(:,d), trainx,ones(size(trainx,1), 1));
mx.meanfunc2= meanfunc2; mx.inffunc2=@infLaplace; mx.covfunc2=covfunc2; mx.likfunc2 =likfunc2; mx.hyp2 = hyp2;
end
end
%混合响应训练
% conmodeldir = '.\MMRGP_subprograms';    %参数变为A   %%！！这个方法模型构建有问题，先别用！！
% conmodeldir = 'F:\学习工作-wzc\搞点研究\混合响应建模\0625混合响应稀疏近似+相关性+多分类尝试\MRSM_subprograms';
% conmodeldir = '.\MRSM_subprograms0';   %原始（超参数为Z0）
% addpath(conmodeldir);
% trainym = trainy; trainym(:,d)=fm;
% [trainym,settings]=mapminmax(trainym',-1,1);
% trainym = trainym';
% mx = MRSM(trainx,trainym);
% mx.settings = settings;
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


