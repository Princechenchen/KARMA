% mROI_SUR2w_varcs02lPD_fast_PI_pil3brd - box3d arm, Branin 2D 版(2026-10-03)
% = pil3br 逐字克隆 + 【早段延迟注入】(针对 16 种子池定位的"早段信任透支"局限:
%   box3 的 gate 自步1即开(α_eff≈0.7), 在裁判尚无数据时先验就接管 ROI 识别,
%   早段真实边界带覆盖仅 13.8%(C7 数据驱动为 45%), ACC 差距中段起扩大)。
%
%   唯一功能差异(相对 pil3br): 前 warmup 步强制 pi3_active=false —— 先验"连参赛
%   资格都没有"(evaluate_fitness_fast_PI3 退回 IL+G+U 三目标, U 即数据驱动的分类
%   熵目标), 让数据驱动信号完成初始测绘; V1 裁判与 α EMA 照常逐步运行(信任在
%   warmup 期间继续积累), 第 warmup+1 步起恢复 box3 原行为(退火场+gate+兑现)。
%   warmup 默认 5, 可由 model.pi3_warmup 覆盖。
%   其余(C 退火/A 启停/B 兑现/V1 裁判/分量 α EMA(0.5)+地板 0.1/max 融合/L2 禁用)
%   逐字承袭 pil3br。
function [x_star, model] = mROI_SUR2w_varcs02lPD_fast_PI_pil3brd(model)

% (pil3brd: 分量独立 α 变体说明承袭 pil3br。% 分量独立 α 变体（2026-08-16）。
%   1) 先验携带方式：model.pi_comp_fns = {π_hg, π_bnd} 两个分量句柄。
%   2) 可信度分量独立：α_hg / α_bnd 各自 EMA(η=0.5)+地板(默认 0.1)，
%      cons_hg=corr(π_hg,连续残差场), cons_bnd=corr(π_bnd,分类残差场)
%      （loo_consistency_uniform_comp，固定 51×51 均匀参考集）。
%   3) 注入用 α 加权 max 融合：fused(x) = max(π_hg(x)^α_hg, π_bnd(x)^α_bnd)
%      L1-a：fused 挂 model.pi_phys_fn 且 pi_alpha=1 传给 _PI3 MOPSO；
%      L2 ：禁用(pil1 语义, metric stays pure M_data)。
%   依赖 model 字段：pi_comp_fns / pi_comp_adaptive / pi_comp_eta /
%     pi_comp_alpha_min / pi_comp_alpha / pi_unif_grid / pi3_warmup(新增, 默认 5)。

ns = size(model.candidate,1); yd = size(model.init_value,2); nts = size(model.testsample,1);
canddata = model.candidate;
testsample = model.testsample;
%% 关键区域分析确定候选集(根据关键区域生成候选样本)
dim = model.dim;
lower_bounds=zeros(1,dim);   % 搜索空间的下界   三组合
upper_bounds=ones(1,dim);   % 搜索空间的上界   三组合
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
fast_options = struct('batch_prediction', true, 'batch_archive', true, ...
    'reuse_sampling_template', true, 'vectorized_roi_metrics', true);

% ===== [box3d 新增] 延迟注入参数 =====
step_now = size(model.now_x,1) - model.n_init + 1;   % 当前注入步(1..n_seq)
warmup = 5;
if isfield(model,'pi3_warmup') && ~isempty(model.pi3_warmup)
    warmup = model.pi3_warmup;
end

% ===== 分量独立 α（EMA on 均匀参考集 分量 consistency，地板 0.1）=====
if isfield(model,'pi_comp_adaptive') && model.pi_comp_adaptive
    eta = 0.5;  if isfield(model,'pi_comp_eta')       && ~isempty(model.pi_comp_eta),       eta  = model.pi_comp_eta;       end
    amin = 0.1; if isfield(model,'pi_comp_alpha_min') && ~isempty(model.pi_comp_alpha_min), amin = model.pi_comp_alpha_min; end
    a_prev = [1 1]; if isfield(model,'pi_comp_alpha') && ~isempty(model.pi_comp_alpha), a_prev = model.pi_comp_alpha; end
    uopts = struct('n_grid', 51);
    if isfield(model,'pi_unif_grid') && ~isempty(model.pi_unif_grid), uopts.n_grid = model.pi_unif_grid; end
    [cons_vec, ~] = loo_consistency_uniform_comp(model, uopts);   % V1 裁判(Branin); 分量独立（不动 rng/代理模型）
    a_new = max(amin, eta*a_prev + (1-eta)*cons_vec);
    model.pi_comp_alpha = a_new;
    % 轨迹记录（护栏初始化）
    if ~isfield(model,'pi_comp_cons_hist')  || isempty(model.pi_comp_cons_hist),  model.pi_comp_cons_hist  = zeros(0,2); end
    if ~isfield(model,'pi_comp_alpha_hist') || isempty(model.pi_comp_alpha_hist), model.pi_comp_alpha_hist = zeros(0,2); end
    model.pi_comp_cons_hist(end+1,:)  = cons_vec;
    model.pi_comp_alpha_hist(end+1,:) = a_new;
    fprintf('[_PI-pil3brd] step %d : α_hg %.3f→%.3f (cons %.3f) | α_bnd %.3f→%.3f (cons %.3f)\n', ...
        size(model.now_x,1), a_prev(1), a_new(1), cons_vec(1), a_prev(2), a_new(2), cons_vec(2));
end

% ===== α 加权 max 融合（L1-a 与 L2 共用）=====
a_use = [1 1]; if isfield(model,'pi_comp_alpha') && ~isempty(model.pi_comp_alpha), a_use = model.pi_comp_alpha; end
fn_hg = model.pi_comp_fns{1};  fn_bnd = model.pi_comp_fns{2};
fused_fn = @(X) max(fn_hg(X).^a_use(1), fn_bnd(X).^a_use(2));

% ===== [box3-C 主] 先验场退火: π_eff(x) = fused(x)·min(1, ρ̂(x)^(-1/L)) =====
knn_k = 5;  L_anneal = 3;  tau_drop = 0.25;
if isfield(model,'pi3_anneal')
    if isfield(model.pi3_anneal,'k'),        knn_k   = model.pi3_anneal.k;         end
    if isfield(model.pi3_anneal,'L'),        L_anneal= model.pi3_anneal.L;         end
    if isfield(model.pi3_anneal,'tau_drop'), tau_drop= model.pi3_anneal.tau_drop;  end
end
Xv = model.now_x;                                   % 已采样点(归一化域, 按值捕获)
pi_eff_fn = @(X) fused_fn(X) .* prior_anneal_factor(X, Xv, knn_k, L_anneal);

% ===== [box3-A 辅 + box3d 延迟] 目标启停: warmup 期强制关闭, 之后按 α_eff 门控 =====
alpha_eff = max(a_use);                             % 场内最可信通道
pi_on = (step_now > warmup) && (alpha_eff >= tau_drop);

% --- L1-a(box3+box3d): 退火场挂 pi_phys_fn, pi3_active/pi3_w_phi 传给 _PI3 MOPSO ---
model.pi_phys_fn = pi_eff_fn;
model.pi_alpha   = 1;
model.pi3_active = pi_on;
model.pi3_w_phi  = alpha_eff;
if ~isfield(model,'pi3_hist') || isempty(model.pi3_hist), model.pi3_hist = zeros(0,4); end
model.pi3_hist(end+1,:) = [a_use, alpha_eff, double(pi_on)];   % [α_hg α_bnd α_eff gate](诊断, 断点续跑不保)
fprintf('[_PI-pil3brd] step %d (warmup=%d) : α_eff=%.3f gate=%d w_phi=%.3f | 退火 k=%d L=%.1f τ=%.2f\n', ...
    size(model.now_x,1), warmup, alpha_eff, pi_on, alpha_eff, knn_k, L_anneal, tau_drop);
% [box3d] warmup 期用基础版 MOPSO(IL+G+U, 与 C1/C7 等 R0 臂同款、久经考验), 而非 PI3 的
%   pi3_active=false 回退路径 —— 后者在 cell4(rs=1) 步1 稳定触发原生堆损坏(0xc0000374,
%   两次复现于 MOPSO 迭代 15/201 附近; 该回退路径历史上从未被长跑检验)。
if pi_on
    occupied_table = mopso_adaptive_elitism_crowding_PI3(dim, bounds, window_size_bounds, ...
        model, num_samples, iterations1, iterations2, num_samples, elitism_size, ...
        model_num, roi_cache, fast_options);
else
    occupied_table = mopso_adaptive_elitism_crowding(dim, bounds, window_size_bounds, ...
        model, num_samples, iterations1, iterations2, num_samples, elitism_size, ...
        model_num, roi_cache, fast_options);
end
model.last_roi_table = occupied_table;   % 【诊断记录】暴露本次识别的关键区域集(含 Phi 目标)，供可视化/存档
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
Im = eye(size(model.ClassPos,2));         % n×n的单位矩阵

distvar = dist(model.candidate,model.now_x'); %distance_weight
distvar = min(distvar,[],2);

% ===== 循环外预算（点2 + 第1/3点）=====
cov_TX=define_covmatrix0(model.MMRGP,model.testsample,model.MMRGP.X);  %cov:testx_X
cov_cc=define_covmatrix0(model.MMRGP,model.candidate(1,:),model.candidate(1,:));  %cov:candx_candx
uFSigmaFu_all=zeros(yd,nts);
for jj=1:nts
    cov_tX_tmp=cov_TX(1+yd*(jj-1):yd*jj,:);
    u_tmp=Im-cov_tX_tmp*SigmaF;
    uFSigmaFu_all(:,jj)=sum(u_tmp*invFSigmaF.*u_tmp,2);
end

pnum = model.total_iter-model.n_init;
pid = size(model.now_x,1)-model.n_init+1;
for i=1:ns
    candx = model.candidate(i,:);
    cov_ct=define_covmatrix0(model.MMRGP,candx,model.testsample);  %cov:candx_testX 大
    cov_cX=define_covmatrix0(model.MMRGP,candx,model.MMRGP.X);  %cov:candx_X 小
    cov_cX_invz = cov_cX*invz;
    scandx = s(1+yd*(i-1):yd*i,:);   %sigma_candx
    invs = inv(scandx);
    % new矩阵部件计算
    F_new = [model.MMRGP.fn;Im];   %F(n+1)
    zmn_new =  [[model.MMRGP.z_mn;cov_cX],[cov_cX';cov_cc]]; %z_(m(n+1)*m(n+1))拼接
    zchol_new=poschol(zmn_new); %chol_SIGMA_new
    SigmaF_new=(zchol_new\(zchol_new'\F_new)); %inv_SIGMA_new*F(n+1)

    FSigmaF_new=F_new'*SigmaF_new;  %F(n+1)'*inv_SIGMA_new*F(n+1)
    FSigmaF_chol2=poschol(FSigmaF_new);    %chol(F(n+1)'*inv_SIGMA_new*F(n+1))
    invFSigmaF_new = FSigmaF_chol2\(FSigmaF_chol2'\eye(size(FSigmaF_chol2)));
    for j=1:nts
        cov_tX = cov_TX(1+yd*(j-1):yd*j,:);
        COV = cov_ct(:,1+yd*(j-1):yd*j) - cov_cX_invz*cov_tX';  %第2点：用预算的 cov_cX_invz
        delta_sigma = sum(COV'*invs.*COV',2);

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
metric = (metric+Cvar'+cd'+distvar');  %距离加和 —— 此即 M_data（归一化数据驱动指标）

% ===== L2 物理先验融合（分量独立 α：S = M_data · fused）=====
pi_at_cand = fused_fn(model.candidate);              % [ns × 1]，max(π_hg^α_hg, π_bnd^α_bnd)
% [pil3] L2 乘法仍禁用(pil1 语义): metric stays pure M_data                    % 对齐 [1 × ns]

[bestvalue,index] = max(metric);
x_star = model.candidate(index,:);
if index > floor(ns/(RoI_num+1))*RoI_num
    model.M_ind(size(model.now_value,1)-size(model.init_value,1)+1)=0;
else
    model.M_ind(size(model.now_value,1)-size(model.init_value,1)+1)=1;
end
model.candidate = canddata;
model.testsample = testsample;
end
