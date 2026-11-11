function [mx] = SOGP(trainy,trainx,ClassPos,dm)
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
for d=1:size(trainy,2)
if ClassPos(d)~=2
    meanfunc = @meanConst; covfunc = @covSEiso; likfunc = @likGauss;
    hyp = struct('mean', 0, 'cov', zeros(1,2), 'lik',1);
    hyp = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, trainx, trainy(:,d));
    mx.meanfunc= meanfunc; mx.inffunc=@infGaussLik; mx.covfunc=covfunc; mx.likfunc =likfunc; mx.hyp = hyp;
else
meanfunc2 = @meanConst; hyp2.mean = 0;
covfunc2 = @covSEard; ell = 1.0; sf = 1.0; hyp2.cov = zeros(1,dm+1);
likfunc2 = @likErf;
hyp2 = struct('mean', 0, 'cov', zeros(1,dm+1));
zeroindex = find(trainy(:,d)~=1);
trainy(zeroindex,d)=-1;
hyp2 = minimize(hyp2, @gp, -2, @infLaplace, meanfunc2, covfunc2, likfunc2, trainx, trainy(:,d));
mx.meanfunc2= meanfunc2; mx.inffunc2=@infLaplace; mx.covfunc2=covfunc2; mx.likfunc2 =likfunc2; mx.hyp2 = hyp2;
end
end
mx.classpos=ClassPos;
