function [x_star, model] = mGlobal_SUR2w_varcs02lPD_fast_piBO(model)
%MGLOBAL_SUR2W_VARCS02LPD_FAST_PIBO Global candidate-pool piBO baseline.
% No ROI search is run; the supplied global candidate/reference pools are
% scored using the unchanged MMRGP data metric and a piBO prior multiplier.
ns = size(model.candidate,1); yd = size(model.init_value,2); nts = size(model.testsample,1);
canddata = model.candidate;
testsample = model.testsample;
if ~isfield(model,'pi_phys_fn') || isempty(model.pi_phys_fn)
    error('piBO:MissingPrior','model.pi_phys_fn is required.');
end
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
roi_cache = prepare_roi_predict_cache(model.MMRGP);
% A3 is numerically equivalent in the fixed-seed tests. Keep A4 available
% for ablation, but leave it off by default because batch archive updates
% can change truncation order and therefore adaptive-restart trajectories.
fast_options = struct('batch_prediction', true, 'batch_archive', true, ...
    'reuse_sampling_template', true, 'vectorized_roi_metrics', true);
occupied_table = table();
model.last_roi_table = occupied_table;   % 【诊断记录】暴露本次识别的关键区域集，供测试脚本可视化/存档（不改行为）
model.candidate = canddata;
model.testsample = testsample;
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
Im = eye(size(model.ClassPos,2));         % n×n的单位矩阵

distvar = dist(model.candidate,model.now_x'); %distance_weight
distvar = min(distvar,[],2);

% ===== 循环外预算（点2 + 第1/3点）=====
% 点2: cov_TX 不依赖候选 i，移出 for-i 循环
cov_TX=define_covmatrix0(model.MMRGP,model.testsample,model.MMRGP.X);  %cov:testx_X
% 第3点: cov_cc 对任意单点为常数（A*A'+扰动），移出 for-i 算一次
cov_cc=define_covmatrix0(model.MMRGP,model.candidate(1,:),model.candidate(1,:));  %cov:candx_candx
% 第1点: uFSigmaFu（旧模型不确定性）不依赖候选 i（仅依赖 cov_TX/SigmaF/invFSigmaF），预算到 for-i 外（nts 次，而非 ns*nts 次）
uFSigmaFu_all=zeros(yd,nts);
for jj=1:nts
    cov_tX_tmp=cov_TX(1+yd*(jj-1):yd*jj,:);
    u_tmp=Im-cov_tX_tmp*SigmaF;
    uFSigmaFu_all(:,jj)=sum(u_tmp*invFSigmaF.*u_tmp,2);
end

pnum = model.total_iter-model.n_init;
pid = size(model.now_x,1)-model.n_init+1;
for i=1:ns
    % 第5点: 去掉逐候选 fprintf（I/O 开销，不影响计算）

    candx = model.candidate(i,:);
    cov_ct=define_covmatrix0(model.MMRGP,candx,model.testsample);  %cov:candx_testX 大
    cov_cX=define_covmatrix0(model.MMRGP,candx,model.MMRGP.X);  %cov:candx_X 小
    % 第3点: cov_cc 已移到循环外（常数）
    % 第2点: cov_cX*invz 不依赖 j，预算到 for-j 外
    cov_cX_invz = cov_cX*invz;
    scandx = s(1+yd*(i-1):yd*i,:);   %sigma_candx
    invs = inv(scandx);
    % new矩阵部件计算
    F_new = [model.MMRGP.fn;Im];   %F(n+1)
    zmn_new =  [[model.MMRGP.z_mn;cov_cX],[cov_cX';cov_cc]]; %z_(m(n+1)*m(n+1))拼接
    zchol_new=poschol(zmn_new); %chol_SIGMA_new
    SigmaF_new=(zchol_new\(zchol_new'\F_new)); %inv_SIGMA_new*F(n+1)

    FSigmaF_new=F_new'*SigmaF_new;  %F(n+1)'*inv_SIGMA_new*F(n+1)
    FSigmaF_chol2=poschol(FSigmaF_new);  %chol(F(n+1)'*inv_SIGMA_new*F(n+1))
    invFSigmaF_new = FSigmaF_chol2\(FSigmaF_chol2'\eye(size(FSigmaF_chol2)));
    for j=1:nts
        cov_tX = cov_TX(1+yd*(j-1):yd*j,:);
        COV = cov_ct(:,1+yd*(j-1):yd*j) - cov_cX_invz*cov_tX';  %第2点：用预算的 cov_cX_invz
        delta_sigma = sum(COV'*invs.*COV',2);

        % 第1点: uFSigmaFu 已预算到 uFSigmaFu_all（不再每 i 每 j 重算）
        cov_tXcandx=[cov_tX,cov_ct(:,1+yd*(j-1):yd*j)];
        u2=Im-cov_tXcandx*SigmaF_new;
        uFSigmaFu_new=sum(u2*invFSigmaF_new.*u2,2);

        delta_beta = uFSigmaFu_all(:,j)-uFSigmaFu_new;  %第1点
        delta_uncertainty = delta_sigma+delta_beta;
        dvar(:,j) = delta_uncertainty;
    end
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
cd= cd./max(cd); cd = mean(cd,2);
Cvar = Cvar./max(Cvar);
metric= metric./max(metric);
distvar = distvar./max(distvar);
metric = (metric+Cvar'+cd'+distvar');  %距离加和

% Match piBO Algorithm 1: n counts sequential calls after the initial D_0.
n_obs = size(model.now_x,1) - size(model.init_x,1) + 1;
n_obs = max(1,n_obs);
beta = (model.total_iter - model.n_init) / 10;
if isfield(model,'pibo_beta') && ~isempty(model.pibo_beta), beta = model.pibo_beta; end
pi_at_cand = model.pi_phys_fn(model.candidate);
if numel(pi_at_cand) ~= ns
    error('piBO:InvalidPrior','model.pi_phys_fn must return one value per candidate.');
end
prior_weight = pibo_prior_weight(pi_at_cand(:), n_obs, beta);
data_score = metric(:);
metric = (data_score .* prior_weight(:))';
model.pibo_last_data_score = data_score;
model.pibo_last_prior = pi_at_cand(:);
model.pibo_last_weight = prior_weight(:);
model.pibo_last_score = metric(:);
model.pibo_last_n = n_obs;
model.pibo_last_exponent = beta / n_obs;

[bestvalue,index] = max(metric);
x_star = model.candidate(index,:);
model.M_ind(size(model.now_value,1)-size(model.init_value,1)+1)=0;
model.candidate = canddata;
model.testsample = testsample;
