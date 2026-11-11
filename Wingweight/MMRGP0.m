function [mx] = MMRGP0(trainy,trainx,ClassPos,dm)
[parent_dir, ~, ~] = fileparts(pwd);
disp ('executing gpml startup script...')
mydir='F:\学习工作\搞点研究\混合响应建模\单一响应建模\';
addpath (mydir);
dirs = {'cov','doc','inf','lik','mean','prior','util'};
for d = dirs
  addpath (fullfile (mydir, d{1}))
end
dirs = {{'util','minfunc'},{'util','minfunc','compiled'}};
for d = dirs
  addpath (fullfile (mydir, d{1}{:}))
end
addpath([mydir,'/util/sparseinv'])
trainym = trainy;
for d = find(ClassPos==2)
meanfunc = @meanConst; hyp.mean = 0;
covfunc = @covSEard; ell = 1.0; sf = 1.0; hyp.cov = zeros(1,dm+1);
likfunc = @likErf;
hyp = struct('mean', 0, 'cov', zeros(1,dm+1));
zeroindex = find(trainy(:,d)~=1);
trainy(zeroindex,d)=-1;
hyp = minimize(hyp, @gp, -40, @infLaplace, meanfunc, covfunc, likfunc, trainx, trainy(:,d));
[ymu,~,fm,~,lp2] = gp(hyp, @infLaplace, meanfunc, covfunc, likfunc, trainx, trainy(:,d), trainx,ones(size(trainx,1), 1));
trainym(:,d)=fm;
end
conmodeldir = [parent_dir,'\MRSM_subprograms1'];
addpath(conmodeldir);
[trainym,settings]=mapminmax(trainym',-1,1);
trainym = trainym';
mx = MRSM(trainx,trainym);
mx.settings = settings;
mx.classpos=ClassPos;
