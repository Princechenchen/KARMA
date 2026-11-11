function [x_star, model] =  mROI_SUR2w_varcs02lPD(model)   %候选集服从关键区域，参考集每次跌代也根据关键区域生成
ns = size(model.candidate,1); yd = size(model.init_value,2); nts = size(model.testsample,1);
canddata = model.candidate; %原始候选集备份
testsample = model.testsample; %原始候选集备份
%% 关键区域分析确定候选集(根据关键区域生成候选样本)
dim = model.dim;
lower_bounds=zeros(1,dim);   % 搜索空间的下界   三组合
upper_bounds=ones(1,dim);   % 搜索空间的上界
bounds = [lower_bounds; upper_bounds]';  % 输入范围
num_samples = 50;  % 每个区域随机抽取的样本数量 80
iterations1 = 200;  % 迭代次数 200
RoI_num=floor(dim*2);%关键区域数量 4 改为维度的一半
iterations2 = round(iterations1/RoI_num);  %重启周期
elitism_size = 10;  % 精英数量
model_num=1;
window_min=0.35;%窗口相对原维度的最小比例
window_max=0.85;%窗口相对原维度的最大比例
window_size_bounds = zeros(dim, 2);
for d = 1:dim
    window_size_bounds(d, :) = [(upper_bounds(d) - lower_bounds(d)) * window_min, ...
        (upper_bounds(d) - lower_bounds(d)) * window_max];
end
occupied_table = mopso_adaptive_elitism_crowding(dim, bounds, window_size_bounds, model, num_samples, iterations1,iterations2, num_samples, elitism_size,model_num);
model.candidate = gen_ROIs(model,ns,occupied_table);
model.testsample = gen_ROIs(model,nts,occupied_table);
ns = size(model.candidate,1); nts = size(model.testsample,1);
%% 逐步前向缩减
[y, s, model.MMRGP]=predict_resp(model.MMRGP,model.candidate);
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
    zmn_new =  [[model.MMRGP.z_mn;cov_cX],[cov_cX';cov_cc]]; %z_(m(n+1)*m(n+1))拼接
    zchol_new=poschol(zmn_new); %chol_SIGMA_new  
    SigmaF_new=(zchol_new\(zchol_new'\F_new)); %inv_SIGMA_new*F(n+1)
%     invz_new = blockm_inv(invz,cov_cX,cov_cc);  %矩阵分块求逆,平替前三行
%     SigmaF_new = invz_new*F_new;

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
distvar = distvar./max(distvar);
metric = (metric+Cvar'+cd'+distvar');  %距离加和


% %   weight
% meandvar = mean(mdvar,2); maxdvar=max(mdvar,[],2);
% weight = 0.5*(meandvar./sum(meandvar)+maxdvar./sum(maxdvar));
% metric =  distvar'.*sum(mdvar.*weight); 

[bestvalue,index] = max(metric);
x_star = model.candidate(index,:);
% [index] = ismember(canddata, x_star, 'rows');  %返回最佳点在原始候选集中的索引
% index = find(index);
% [index0] = ismember(model.candidate0, x_star, 'rows');  %返回最佳点在原始候选集中的索引
% index0 = find(index0);
% model.M_ind(size(model.now_value,1)-size(model.init_value,1)+1)=index;
% canddata(index,:)=[];
if index > floor(ns/(RoI_num+1))*RoI_num %判别所选点是否位于关键区域中
    model.M_ind(size(model.now_value,1)-size(model.init_value,1)+1)=0;
else 
    model.M_ind(size(model.now_value,1)-size(model.init_value,1)+1)=1;
end
model.candidate = canddata;
model.testsample = testsample;