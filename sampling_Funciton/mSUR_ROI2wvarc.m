function [x_star, model] =  mSUR_ROI2wvarc(model)   %增加了方差预测中，对均值不确定性的表征项（后面的一大坨）%三个响应的预测误差归一化求和来综合
yd = size(model.init_value,2); nts = size(model.testsample,1);
canddata = model.candidate; %原始候选集备份
testsample = model.testsample; %原始参考集备份
%%%%% ROI %%%%%
% 将存放partitioned_gp.py的文件夹路径添加到Python搜索路径
py_path = py.sys.path; [func_dir, ~, ~] = fileparts(mfilename('fullpath')); parent_dir = fileparts(func_dir);
folder_path = fullfile(parent_dir, 'ROImethod');
insert(py_path, int32(0), folder_path);  % 添加到路径最前面，优先搜索
% 1. 导入Python模块（若文件名为partitioned_gp.py，则模块名为partitioned_gp）
pgp_module = py.importlib.import_module('partitioned_gp');
ROIclf_entity = pgp_module.pal_region_classifier(n_partitions=2, seed=py.None, n_iter=int32(300)); %"F:\tools\Python39\python.exe" -m pip install scikit-learn
% 2. 对每个响应进行计算对应的ROI
% 转换成python数据类型   %注意所有整数都涉及数据类型的转变--->int32() [x,id] = unique(model.now_x,"rows"); y=model.now_value(id,:);
X_init_py = py.numpy.array(model.now_x);
candx = py.numpy.array(model.candidate);
testx = py.numpy.array(model.testsample);
for i=1:length(model.ClassPos)
    Z_init_py = py.numpy.array(model.now_value(:,i));
    % ROI关键区域分类
    try 
        ROIclf_entity.fit(X_init_py, Z_init_py);
    catch
        if i==1
            ROIclf_entity = model.entity{i};
        else
        ROIclf_entity = model.entity{i-1};
        end
        fprintf('第%d样本, 第%d响应, 继承分类器 \n', size(model.now_x,1)-size(model.init_x,1)+1, i-1);
    end
    % 预测类别
    C_cand = ROIclf_entity.predict(candx);
    CR_cand(i,:) = double(C_cand);
    C_test = ROIclf_entity.predict(testx);
    CR_test(i,:) = double(C_test);
    model.entity{i} = ROIclf_entity;
end
% 确定异构区域（基于梯度）
cand_labels = unique(CR_cand); cand_labels = cand_labels';
test_labels = unique(CR_test); test_labels = test_labels';
% 根据预测方差大小选定各响应关键区域(不确定性最大的)  %%选择关键区域的方式有待考虑%%
[~, s,model.MMRGP]=predict_resp(model.MMRGP,model.testsample);
for i=1:yd
    var(:,i) = s(i:yd:end,i);
end
var = mapminmax(var', 0, 1); var = var';
sumvar=sum(var,2);
for c = test_labels
    for i=1:yd
        ind = CR_test(i,:)==c;
        indval(c+1,i) = sum(sumvar(ind));
    end
end
[~,ROIid]=max(indval);
% 权衡每个响应的关键区域  %%如何有效综合多个关键区域？%% 暂定23交集
ind = CR_cand'==cand_labels(ROIid); ind = ind';%连续响应取方差大的
candidateX = model.candidate(sum(ind)>=yd-1,:);
model.candidate = candidateX;

ind = CR_test'==test_labels(ROIid); ind = ind';
RefX = model.testsample(sum(ind)>=yd-1,:);
model.testsample = RefX;
    
ns = size(model.candidate,1); nts = size(model.testsample,1);

%% 逐步前向缩减
[y, s, model.MMRGP]=predict_resp(model.MMRGP,model.candidate);
var=[];
for i=1:yd
    var(:,i) = s(i:yd:end,i);
end
SUR_metric=[];
zchol=poschol(model.MMRGP.z_mn); %chol_SIGMA
invz = zchol\(zchol'\eye(size(zchol))); %inv_SIGMA
SigmaF=(zchol\(zchol'\model.MMRGP.fn));  %inv_SIGMA*F
FSigmaF=model.MMRGP.fn'*SigmaF;   %F*inv_SIGMA*F
FSigmaF_chol=poschol(FSigmaF);    %chol(F*inv_SIGMA*F)
invFSigmaF = FSigmaF_chol\(FSigmaF_chol'\eye(size(FSigmaF_chol)));
% Im = ones(model.MMRGP.m,1);  
Im = eye(size(model.ClassPos,2));         % n×n的单位矩阵

distvar = dist(model.candidate,model.now_x'); %distance_weight
distvar = min(distvar,[],2);

pnum = model.total_iter-model.n_init;
pid = size(model.now_x,1)-model.n_init+1;
for i=1:ns
    fprintf('第%d/%d采样点, 当前迭代 %d/%d \n', pid, pnum, i, ns);
    
    
    candx = model.candidate(i,:);
%     cov_ct=define_covmatrix(model.MMRGP,candx,model.testsample);  %cov:candx_testX 大
%     cov_cX=define_covmatrix(model.MMRGP,candx,model.MMRGP.X);  %cov:candx_X 小
%     cov_cc=define_covmatrix(model.MMRGP,candx,candx);  %cov:candx_candx
    cov_ct=define_covmatrix0(model.MMRGP,candx,model.testsample);  %cov:candx_testX 大
    cov_cX=define_covmatrix0(model.MMRGP,candx,model.MMRGP.X);  %cov:candx_X 小
    cov_cc=define_covmatrix0(model.MMRGP,candx,candx);  %cov:candx_candx
    cov_TX=define_covmatrix0(model.MMRGP,model.testsample,model.MMRGP.X);  %cov:testx_X 
    scandx = s(1+yd*(i-1):yd*i,:);   %sigma_candx
%     schol = chol(scandx);
%     invs = schol\(schol'\eye(size(schol)));
    invs = inv(scandx);
    delta_sigma=0; %delta_sigma=[];
    % new矩阵部件计算
    F_new = [model.MMRGP.fn;Im];   %F(n+1)
%     zmn_new =  [[model.MMRGP.z_mn;cov_cX],[cov_cX';cov_cc]]; %z_(m(n+1)*m(n+1))拼接
%     zchol_new=poschol(zmn_new); %chol_SIGMA_new  
%     SigmaF_new=(zchol_new\(zchol_new'\F_new)); %inv_SIGMA_new*F(n+1)
    invz_new = blockm_inv(invz,cov_cX,cov_cc);  %矩阵分块求逆,平替前三行
    SigmaF_new = invz_new*F_new;

    FSigmaF_new=F_new'*SigmaF_new;  %F(n+1)'*inv_SIGMA_new*F(n+1)
    FSigmaF_chol2=poschol(FSigmaF_new);  %chol(F(n+1)'*inv_SIGMA_new*F(n+1))
    invFSigmaF_new = FSigmaF_chol2\(FSigmaF_chol2'\eye(size(FSigmaF_chol2)));
%     uc = Im-cov_cX*SigmaF; %  uc=Im-cov(candx,x)*inv_SIGMA*F       %1
    for j=1:nts
        % uncertainty of predict var
%         testx = model.testsample(j,:);
%         cov_tX=define_covmatrix(model.MMRGP,testx,model.MMRGP.X);  %cov:testx_X
        cov_tX = cov_TX(1+yd*(j-1):yd*j,:);
%         u=Im-cov_tX*SigmaF; %  u=Im-cov(testx,x)*inv_SIGMA*F       %1
%         FSigmaFu = FSigmaF_chol\(FSigmaF_chol'\u');                %1  
        COV = cov_ct(:,1+yd*(j-1):yd*j) - cov_cX*invz*cov_tX'; %COV: candx_testx cov_cX*(zchol\(zchol'\cov_tX'))!!这个地方可能需要补上后面一坨%+uc*FSigmaFu
        delta_sigma = sum(COV'*invs.*COV',2);  %sigma_n+1(textx)-sigma_n(textx)
%         delta_sigma = COV'*invs*COV;  %sigma_n+1(textx)-sigma_n(textx)

        % uncertainty of predict mean
        % old
%         uFSigmaFu=u*FSigmaFu;  % u*inv(F*inv_SIGMA*F)*u'           %1
        u=Im-cov_tX*SigmaF; %  u=Im-cov(testx,x)*inv_SIGMA*F
        uFSigmaFu=sum(u*invFSigmaF.*u,2);  % u*inv(F*inv_SIGMA*F)*u'
%         uFSigmaFu=u*(FSigmaF_chol\(FSigmaF_chol'\u'));  % u*inv(F*inv_SIGMA*F)*u'

        % new
        cov_tXcandx=[cov_tX,cov_ct(:,1+yd*(j-1):yd*j)]; %cov:testx_X+candx 拼接
        u2=Im-cov_tXcandx*SigmaF_new; % u2=Im-cov(testx,x+candx)*inv_SIGMA_new*F(n+1)
        uFSigmaFu_new=sum(u2*invFSigmaF_new.*u2,2);
%         uFSigmaFu_new=u2*(FSigmaF_chol2\(FSigmaF_chol2'\u2'));  % u2*inv(F(n+1)*inv_SIGMA_new*F(n+1))*u2'
        
        delta_beta = uFSigmaFu-uFSigmaFu_new;   %
        delta_uncertainty = delta_sigma+delta_beta;
        dvar(:,j) = delta_uncertainty;
%         dvar(:,j) = diag(delta_uncertainty); %??? 是否加和，是否只取对角线
%         dsigma(:,j) = diag(delta_sigma);
%         dbeta(:,j) = diag(delta_beta);
    end
%     SUR_metric(:,:,i)=delta_uncertainty;
%     mdvar(:,i) = sum(dvar,2);

%     % none weight
%     metric(i) = sum(sum(dvar));
    % dist weight(0.5mean+0.5max)
    meandvar = mean(dvar,2); maxdvar=max(dvar,[],2);
    weight = 0.5*(meandvar./sum(meandvar)+maxdvar./sum(maxdvar));
    metric(i) = sum(sum(dvar.*weight));
end
% predict variance weight
var(:,model.ClassPos==2) = var(:,model.ClassPos==2);
meanvar = mean(var); maxvar=max(var);
weight = 0.5*(meanvar./sum(meanvar)+maxvar./sum(maxvar));
Cvar = sum(var.*weight,2);
cd = 1./exp(abs(model.MMRGP.fc));
% distance weight+ variance weight
% cd= (cd-min(cd))./(max(cd)-min(cd));
% Cvar = (Cvar-min(Cvar))./(max(Cvar)-min(Cvar));
% metric= (metric-min(metric))./(max(metric)-min(metric));
% cd= zscore(cd);
% Cvar = zscore(Cvar);
% metric= zscore(metric);
cd= cd./max(cd); cd = mean(cd,2);
Cvar = Cvar./max(Cvar);
metric= metric./max(metric);
metric = (metric+Cvar'+cd'+distvar');
% metric = (metric+Cvar'+cd').*distvar';

% %   weight
% meandvar = mean(mdvar,2); maxdvar=max(mdvar,[],2);
% weight = 0.5*(meandvar./sum(meandvar)+maxdvar./sum(maxdvar));
% metric =  distvar'.*sum(mdvar.*weight); 

[bestvalue,index] = max(metric);
x_star = model.candidate(index,:);
[index] = ismember(canddata, x_star, 'rows');  %返回最佳点在原始候选集中的索引
index = find(index);
[index0] = ismember(model.candidate0, x_star, 'rows');  %返回最佳点在原始候选集中的索引
index0 = find(index0);
model.M_ind(size(model.now_value,1)-size(model.init_value,1)+1)=index0;
canddata(index,:)=[];
model.candidate = canddata;
model.testsample = testsample;  %还原参考集
% 一点考虑：是否要将选中的候选点从候选集中删除
