function [x_star, model] =  EIER(model)
this_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(this_dir, 'EIER_code'));
canddata = model.candidate; %原始候选集备份
ns = size(model.candidate,1); yd = size(model.init_value,2); nts = size(model.testsample,1);
[y, s, model.MMRGP]=predict_resp(model.MMRGP,model.candidate);
for i=1:yd
    var(:,i) = s(i:yd:end,i);
end
yc = model.MMRGP.fc;
varc = var(:,model.ClassPos==2);
U = abs(yc)./varc;
y(:,model.ClassPos==2) = model.MMRGP.fc;
y = reshape(y',model.MMRGP.m*size(model.candidate,1),1);  
pc = find(model.ClassPos==2);

NGPR=1000;
SUR_metric=[];
zchol=poschol(model.MMRGP.z_mn); %chol_SIGMA
InvK = zchol\(zchol'\eye(size(zchol)));
GPRSamp2 = GPRSampling(model.MMRGP,InvK,model.MMRGP.X,model.MMRGP.Y,model.candidate,NGPR);%%sampling from the GPR mopdel

metric = zeros(ns,1);
for i=1:ns
    i
    
    candx = model.candidate(i,:);
    r_Train_Xi = define_covmatrix0(model.MMRGP,model.MMRGP.X,candx);  %cov:X_candx 训练样本和第i个样本之间的协方差向量
    r_XSamp_Xi = define_covmatrix0(model.MMRGP,model.candidate,candx); %cov:candX_candx 候选样本和第i个样本间的协方差向量
    r_Xi_Xi = model.MMRGP.hyper.A*model.MMRGP.hyper.A';   %sigma_candx 第i个样本得自协方差
    Denominator = r_Xi_Xi - r_Train_Xi'*InvK*r_Train_Xi;
    K11 = InvK + InvK * (r_Train_Xi/(Denominator)*r_Train_Xi')*InvK;
    K12 = -InvK*r_Train_Xi/Denominator;
    K22 = inv(Denominator);
    InvKStar = [K11,K12;K12',K22];%第i个点加入训练集后Gram矩阵的逆
    r_XSamp_Xtrain=define_covmatrix0(model.MMRGP,model.candidate,model.MMRGP.X);  %cov:candX_X  训练样本和测试样本间的协方差向量
    r_updated = [r_XSamp_Xtrain,r_XSamp_Xi];
    PostVar = abs(repmat(diag(r_Xi_Xi),size(model.candidate,1),1) - diag(r_updated*InvKStar*r_updated'));%将第i个样本加入训练样本集后各备选样本的方差
    PostMean = repmat(model.MMRGP.beta,size(model.candidate,1),NGPR) + r_updated*InvKStar*([repmat(model.MMRGP.ymn,1,NGPR);GPRSamp2((i-1)*model.MMRGP.m+1:i*model.MMRGP.m,:)]-repmat(model.MMRGP.beta,(model.MMRGP.n+1),NGPR));
    ProbDecSamp = mean(max(repmat(normcdf(-U),1,NGPR) - normcdf(-abs(PostMean(pc:yd:end,:))./repmat(sqrt(PostVar(pc:yd:end)),1,NGPR)),zeros(ns,NGPR)),1);%总体判断错误率减小值的样本
    metric(i) = mean(ProbDecSamp);%mean(max(ProbDecSamp,zeros(1,NGPR)));  
end


[bestvalue,index] = max(metric);
x_star = model.candidate(index,:);
[index0] = ismember(model.candidate0, x_star, 'rows');  %返回最佳点在原始候选集中的索引
index0 = find(index0);
model.M_ind(size(model.now_value,1)-size(model.init_value,1)+1)=index0(1);
canddata(index,:)=[];
model.candidate = canddata;
% model.M_ind(size(model.now_value,1)-size(model.init_value,1)+1)=index;  %ship案例需要这个，且需要注释上面的7行%%