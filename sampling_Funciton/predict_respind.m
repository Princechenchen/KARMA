function [y, s, model]=predict_respind(model,x)

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

ClassPos = model.ClassPos;
trainx = model.now_x; trainy = model.now_value;
predy = ones(size(x,1),size(trainy,2)); preds = predy;
for d=1:size(trainy,2)
if ClassPos(d)~=2
    meanfunc = model.SOGP.meanfunc; covfunc = model.SOGP.covfunc; likfunc = model.SOGP.likfunc;              
    hyp = model.SOGP.hyp;
    [prey s2, ~, ~, lp, post] = gp(hyp, model.SOGP.inffunc, meanfunc, covfunc, likfunc, trainx, trainy(:,d), x);
    predy(:,d) = prey; preds(:,d) = s2;
else
%分类潜变量求解
% d = find(ClassPos==2);
meanfunc2 = model.SOGP.meanfunc2; covfunc2 = model.SOGP.covfunc2; likfunc2 = model.SOGP.likfunc2; 
hyp2 = model.SOGP.hyp2;
zeroindex = find(trainy(:,d)~=1);  %分类响应0替换为-1 %feval(covfunc)
trainy(zeroindex,d)=-1;         
[ymu,~,fm,fs2,lp2] = gp(hyp2, @infLaplace, meanfunc2, covfunc2, likfunc2, trainx, trainy(:,d), x,ones(size(x,1), 1));
p = exp(lp2);   prey = 2*p-1;    prey(prey<=0) = 0; prey(prey>0) = 1;
predy(:,d) = prey; preds(:,d) = fs2;
end
end

y=predy; s=preds;
end