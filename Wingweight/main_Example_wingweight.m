clear
[parent_dir, ~, ~] = fileparts(pwd);
addpath([parent_dir,'/src']); addpath([parent_dir,'/sampling_Funciton']); addpath([parent_dir,'/testFunc']); addpath([parent_dir,'\grid']);
addpath(parent_dir);
ns = 500; model.ns=ns;
model.variables.ns=ns;
model.variables.dim=10;
model.ClassPos = [1,1,2];
obj_fct = @(x)wingweight(x);
model.total_iter = 50;
model.n_init=30;
model.dim=model.variables.dim;
rerun_num = 10;
rsl=randperm(ns,rerun_num);
for j=1:rerun_num
rs=rsl(j); model.rs=rs;
rng(rs)
model=cand_gen(model,ns,'uniform');
model.testsample = lhsdesign(12000,model.variables.dim);
model.testset = lhsdesign(5000,model.variables.dim);
model.candidate0 = model.candidate;
ytest = obj_fct(model.testset);
my=mean(ytest); my(model.ClassPos==2)=1; ytest=ytest./my;
model.init_x = lhsdesign(model.n_init,model.dim);
model.init_value=obj_fct(model.init_x);
model.init_value = model.init_value./my;
model.MMRGP = MMRGP0(model.init_value,model.init_x,model.ClassPos,model.dim);
model.now_value=model.init_value; model.now_x=model.init_x;
modelmaxMSE{j} = model;
modelmeanMSE{j} = model;
modelIMSE{j} = model;
modelSCF{j} = model;
modelSUR2wvarc{j} = model;
modelIRS{j} = model;
modelROI_SURwvarclPD{j} = model;
modelSUR_ROI2wvarc{j} = model;
modelmemeroy{j} = model;
for i = 1:model.total_iter-model.n_init
    [x_star, modelmaxMSE{j}] = maxMSE(modelmaxMSE{j});
    f_star = obj_fct(x_star); f_star = f_star./my;
    modelmaxMSE{j}.now_value=[modelmaxMSE{j}.now_value;f_star];
    modelmaxMSE{j}.now_x=[modelmaxMSE{j}.now_x;x_star];
    modelmaxMSE{j}.MMRGP = MMRGP0(modelmaxMSE{j}.now_value,modelmaxMSE{j}.now_x,model.ClassPos,model.dim);
    [ypred]=predict_resp(modelmaxMSE{j}.MMRGP,model.testset);
    errormaxMSE = error_func(ytest,ypred,model.ClassPos);
    modelmaxMSE{j}.error{i} = errormaxMSE;
end
for i = 1:model.total_iter-model.n_init
    [x_star, modelmeanMSE{j}] = meanMSE(modelmeanMSE{j});
    f_star = obj_fct(x_star); f_star = f_star./my;
    modelmeanMSE{j}.now_value=[modelmeanMSE{j}.now_value;f_star];
    modelmeanMSE{j}.now_x=[modelmeanMSE{j}.now_x;x_star];
    modelmeanMSE{j}.MMRGP = MMRGP0(modelmeanMSE{j}.now_value,modelmeanMSE{j}.now_x,model.ClassPos,model.dim);
    [ypred]=predict_resp(modelmeanMSE{j}.MMRGP,model.testset);
    errormeanMSE = error_func(ytest,ypred,model.ClassPos);
    modelmeanMSE{j}.error{i} = errormeanMSE;
end
for i = 1:model.total_iter-model.n_init
    [x_star, modelIMSE{j}] = IMSEs(modelIMSE{j});
    f_star = obj_fct(x_star); f_star = f_star./my;
    modelIMSE{j}.now_value=[modelIMSE{j}.now_value;f_star];
    modelIMSE{j}.now_x=[modelIMSE{j}.now_x;x_star];
    modelIMSE{j}.MMRGP = MMRGP0(modelIMSE{j}.now_value,modelIMSE{j}.now_x,model.ClassPos,model.dim);
    [ypred]=predict_resp(modelIMSE{j}.MMRGP,model.testset);
    errorIMSE = error_func(ytest,ypred,model.ClassPos);
    modelIMSE{j}.error{i} = errorIMSE;
end
addpath([parent_dir,'\MS-PSO']);
for i = 1:model.total_iter-model.n_init
    [x_star, modelROI_SURwvarclPD{j}] = mROI_SUR2w_varcs02lPD(modelROI_SURwvarclPD{j});
    f_star = obj_fct(x_star);  f_star = (f_star-my)./ sy;
    modelROI_SURwvarclPD{j}.now_value=[modelROI_SURwvarclPD{j}.now_value;f_star];
    modelROI_SURwvarclPD{j}.now_x=[modelROI_SURwvarclPD{j}.now_x;x_star];
    modelROI_SURwvarclPD{j}.MMRGP = MMRGP0(modelROI_SURwvarclPD{j}.now_value,modelROI_SURwvarclPD{j}.now_x,model.ClassPos,model.dim);
    [ypred]=predict_resp(modelROI_SURwvarclPD{j}.MMRGP,model.testset);
    errorROI_SURwvarclPD = error_func(ytest,ypred,model.ClassPos);
    modelROI_SURwvarclPD{j}.error{i} = errorROI_SURwvarclPD;
end
modelSCF{j}.SOGP = SOGP(model.init_value,model.init_x,model.ClassPos,model.dim);
for i = 1:model.total_iter-model.n_init
    [x_star, modelSCF{j}] = SCFind(modelSCF{j});
    f_star = obj_fct(x_star); f_star = f_star./my;
    modelSCF{j}.now_value=[modelSCF{j}.now_value;f_star];
    modelSCF{j}.now_x=[modelSCF{j}.now_x;x_star];
    modelSCF{j}.SOGP = SOGP(modelSCF{j}.now_value,modelSCF{j}.now_x,model.ClassPos,model.dim);
    [ypred]=predict_respind(modelSCF{j},model.testset);
    errorSCF = error_func(ytest,ypred,model.ClassPos);
    modelSCF{j}.error{i} = errorSCF;
end
for i = 1:model.total_iter-model.n_init
    [x_star, modelIRS{j}] = AK_MCS_IRS2(modelIRS{j},obj_fct);
    f_star = obj_fct(x_star); f_star = f_star./my;
    modelIRS{j}.now_value=[modelIRS{j}.now_value;f_star];
    modelIRS{j}.now_x=[modelIRS{j}.now_x;x_star];
    modelIRS{j}.MMRGP = MMRGP0(modelIRS{j}.now_value,modelIRS{j}.now_x,model.ClassPos,model.dim);
    [ypred]=predict_resp(modelIRS{j}.MMRGP,model.testset);
    errorIRS = error_func(ytest,ypred,model.ClassPos);
    modelIRS{j}.error{i} = errorIRS;
end
for i = 1:model.total_iter-model.n_init
    [x_star, modelSUR_ROI2wvarc{j}] = mSUR_ROI2wvarc(modelSUR_ROI2wvarc{j});
    f_star = obj_fct(x_star);  f_star = f_star./my;
    modelSUR_ROI2wvarc{j}.now_value=[modelSUR_ROI2wvarc{j}.now_value;f_star];
    modelSUR_ROI2wvarc{j}.now_x=[modelSUR_ROI2wvarc{j}.now_x;x_star];
    modelSUR_ROI2wvarc{j}.MMRGP = MMRGP0(modelSUR_ROI2wvarc{j}.now_value,modelSUR_ROI2wvarc{j}.now_x,model.ClassPos,model.dim);
    [ypred]=predict_resp(modelSUR_ROI2wvarc{j}.MMRGP,model.testset);
    errorSUR_ROI2wvarc = error_func(ytest,ypred,model.ClassPos);
    modelSUR_ROI2wvarc{j}.error{i} = errorSUR_ROI2wvarc;
end
end
