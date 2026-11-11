function mx = MMRGP0_crimes(trainy,trainx,ClassPos,dm,gpml_root)
% Portable MMRGP0_correxp wrapper: original numerics, configurable GPML path.
assert(isfolder(gpml_root),'GPML root does not exist: %s',gpml_root);
addpath(gpml_root);
for d = {'cov','doc','inf','lik','mean','prior','util'}
    addpath(fullfile(gpml_root,d{1}));
end
for d = {{'util','minfunc'},{'util','minfunc','compiled'}}
    addpath(fullfile(gpml_root,d{1}{:}));
end
addpath(fullfile(gpml_root,'util','sparseinv'));

trainym = trainy;
for d = find(ClassPos==2)
    meanfunc = @meanConst;
    covfunc = @covSEard;
    likfunc = @likErf;
    hyp = struct('mean',0,'cov',zeros(1,dm+1));
    zeroindex = find(trainy(:,d)~=1);
    trainy(zeroindex,d) = -1;
    hyp = minimize(hyp,@gp,-40,@infLaplace,meanfunc,covfunc,likfunc,trainx,trainy(:,d));
    [~,~,fm,~,~] = gp(hyp,@infLaplace,meanfunc,covfunc,likfunc, ...
        trainx,trainy(:,d),trainx,ones(size(trainx,1),1));
    trainym(:,d) = fm;
end

adapter_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(adapter_dir));
addpath(fullfile(project_root,'numeric_sample'));
addpath(fullfile(project_root,'MRSM_subprograms1'));
[trainym,settings] = mapminmax(trainym',-1,1);
mx = MRSM_correxp(trainx,trainym');
mx.settings = settings;
mx.classpos = ClassPos;
end
