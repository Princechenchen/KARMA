% mROI_SUR2w_varcs02lPD_fast_PI_pil3_dim - box3dim arm(2026-10-04)
% = pil3 逐字克隆, 唯一功能差异: RoI_num floor(dim*2)=12 → floor(dim)=6
% (iterations2 随式自动 17→33)。单因素对照, 先例=ship v1box3u_dim。
% mROI_SUR2w_varcs02lPD_fast_PI_pil3 - box3 arm (2026-08-26)
% = pil1(L1-only) + 给 L1 的 α 装上真牙(三处增强, 详见 mopso_*_PI3 / prior_anneal_factor):
%   [C 主] 先验场退火: pi_phys_fn = fused .* min(1, ρ̂^(-1/L)), ρ̂ = kNN 局部密度/
%          平均密度(prior_anneal_factor) —— 超采区先验退休、处女地全额; 场随
%          now_x 每步演化 → 区域间排序真实改变 → 支配/拥挤度/兑现全链自动尊重
%          (Pareto 保序不变性被武器化, 而非在幅值上对抗)。
%   [A 辅] 目标启停: α_eff=max(α_hg,α_bnd) < τ_drop=0.25 → 当步 MOPSO 回 3 目标。
%   [B 辅] 兑现加权: commit top-k 均匀随机→质量分加权随机(数据列等权, Φ 列
%          权重=α_eff); 拥挤度 inf+2d 暂不修改。
% L2 乘法仍禁用(pil1 语义): 本臂只考察"先验进 ROI 识别"的方法效果。
% uconf 裁判 / 分量 α EMA / 先验几何与 boxL1 逐字相同。
function [x_star, model] = mROI_SUR2w_varcs02lPD_fast_PI_pil3_dim(model)

% bnd 通道换靶场变体（2026-08-25, 原 _comp 及更早变体全部零改动保留）。
% 与 _comp 的区别仅一处：第 58 行一致性调用换成 loo_consistency_uniform_comp_uconf
%   —— cons_hg 仍=corr(π_hg, 连续 LOO 残差场)（同 _comp）；
%   —— cons_bnd 改=corr(π_bnd, 分类不确定性场 4·p̂(1−p̂))（0825 三探针定谳：
%      LOO 分类残差场对边界信息结构性失明, oracle 脊也 ≤+0.02; uconf 靶场下
%      现行薄壳 9/9 态过 +0.25 线, 均值 +0.447。详见实验先验设置记录 §5.6.1）。
% 其余逻辑与 _comp 逐行相同（分量独立 EMA α、max 融合、L1-a/L2 注入）。% 分量独立 α 变体（2026-08-16, 原 _fast_PI / _fastLOO / _bn / _unif / _unifd 全部不动）。
% 与 _unifd 的区别（其余逻辑逐行相同）：
%   1) 先验携带方式：model.pi_comp_fns = {π_hg, π_bnd} 两个分量句柄
%      （π_hg=难拟合区域/核, π_bnd=边界条带），替代单一 merged π。
%   2) 可信度分量独立：α_hg / α_bnd 各自 EMA(η=0.5)+地板(默认 0.1)，
%      cons_hg=corr(π_hg,连续残差场), cons_bnd=corr(π_bnd,分类残差场)
%      （loo_consistency_uniform_comp，固定 51×51 均匀参考集）——分量各对其
%      目标负责，解决 merged-π 单 α 的"连坐"问题（18 臂总表确诊）。
%   3) 注入用 α 加权 max 融合（不用乘积：另一分量 0.01 地板会压死孤立分量）：
%      fused(x) = max(π_hg(x)^α_hg, π_bnd(x)^α_bnd)
%      L1-a：fused 挂 model.pi_phys_fn 且 pi_alpha=1 传给现有 _PI MOPSO
%            （evaluate_fitness_fast_PI 等文件不动；注意 L1 语义从
%            (mean_R π)^α 变为 mean_R(fused)，属分量化的固有差异，已记录）；
%      L2 ：S(x) = M_data(x) .* fused(x)。
%
%   依赖 model 字段：
%     model.pi_comp_fns        {π_hg, π_bnd}（必需）
%     model.pi_comp_adaptive   true 时启用分量 α 更新（预期用法）
%     model.pi_comp_eta        EMA 系数（默认 0.5）
%     model.pi_comp_alpha_min  地板（默认 0.1）
%     model.pi_comp_alpha      [α_hg, α_bnd]（默认 [1 1]）
%     model.pi_unif_grid       参考集每维网格数（可选，默认 51）
%   不改动代理模型(MMRGP/trend_fun/MRSM)，不改动 gen_ROIs。

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
RoI_num=floor(dim);%box3dim 单因素改动(2026-10-04): floor(dim*2)=12 → floor(dim)=6(6D), iterations2=round(200/6)=33; 先例=ship v1box3u_dim
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

% ===== 分量独立 α（EMA on 均匀参考集 分量 consistency，地板 0.1）=====
if isfield(model,'pi_comp_adaptive') && model.pi_comp_adaptive
    eta = 0.5;  if isfield(model,'pi_comp_eta')       && ~isempty(model.pi_comp_eta),       eta  = model.pi_comp_eta;       end
    amin = 0.1; if isfield(model,'pi_comp_alpha_min') && ~isempty(model.pi_comp_alpha_min), amin = model.pi_comp_alpha_min; end
    a_prev = [1 1]; if isfield(model,'pi_comp_alpha') && ~isempty(model.pi_comp_alpha), a_prev = model.pi_comp_alpha; end
    uopts = struct('n_grid', 51);
    if isfield(model,'pi_unif_grid') && ~isempty(model.pi_unif_grid), uopts.n_grid = model.pi_unif_grid; end
    [cons_vec, ~] = loo_consistency_uniform_comp_uconf(model, uopts);   % 分量独立（不动 rng/代理模型）
    a_new = max(amin, eta*a_prev + (1-eta)*cons_vec);
    model.pi_comp_alpha = a_new;
    % 轨迹记录（护栏初始化）
    if ~isfield(model,'pi_comp_cons_hist')  || isempty(model.pi_comp_cons_hist),  model.pi_comp_cons_hist  = zeros(0,2); end
    if ~isfield(model,'pi_comp_alpha_hist') || isempty(model.pi_comp_alpha_hist), model.pi_comp_alpha_hist = zeros(0,2); end
    model.pi_comp_cons_hist(end+1,:)  = cons_vec;
    model.pi_comp_alpha_hist(end+1,:) = a_new;
    fprintf('[_PI-pil3] step %d : α_hg %.3f→%.3f (cons %.3f) | α_bnd %.3f→%.3f (cons %.3f)\n', ...
        size(model.now_x,1), a_prev(1), a_new(1), cons_vec(1), a_prev(2), a_new(2), cons_vec(2));
end

% ===== α 加权 max 融合（L1-a 与 L2 共用）=====
a_use = [1 1]; if isfield(model,'pi_comp_alpha') && ~isempty(model.pi_comp_alpha), a_use = model.pi_comp_alpha; end
fn_hg = model.pi_comp_fns{1};  fn_bnd = model.pi_comp_fns{2};
fused_fn = @(X) max(fn_hg(X).^a_use(1), fn_bnd(X).^a_use(2));

% ===== [box3-C 主] 先验场退火: π_eff(x) = fused(x)·min(1, ρ̂(x)^(-1/L)) =====
%   ρ̂ = kNN 局部密度/空间平均密度(prior_anneal_factor)。被超额开采的区域
%   先验按对数尺度退火(密度每翻 2^L 倍先验减半), 处女地全额。每步用当前
%   now_x 重建句柄(按值捕获), 场随数据演化。
knn_k = 5;  L_anneal = 3;  tau_drop = 0.25;
if isfield(model,'pi3_anneal')
    if isfield(model.pi3_anneal,'k'),        knn_k   = model.pi3_anneal.k;         end
    if isfield(model.pi3_anneal,'L'),        L_anneal= model.pi3_anneal.L;         end
    if isfield(model.pi3_anneal,'tau_drop'), tau_drop= model.pi3_anneal.tau_drop;  end
end
Xv = model.now_x;                                   % 已采样点(归一化域, 按值捕获)
pi_eff_fn = @(X) fused_fn(X) .* prior_anneal_factor(X, Xv, knn_k, L_anneal);

% ===== [box3-A 辅] 目标启停 + [box3-B 辅] 兑现权重: 统一由 α_eff=max(α_hg,α_bnd) 驱动 =====
alpha_eff = max(a_use);                             % 场内最可信通道
pi_on = alpha_eff >= tau_drop;                      % 连地板都够不着 → 当步回 3 目标

% --- L1-a(box3): 退火场挂 pi_phys_fn, pi3_active/pi3_w_phi 传给 _PI3 MOPSO ---
model.pi_phys_fn = pi_eff_fn;
model.pi_alpha   = 1;
model.pi3_active = pi_on;
model.pi3_w_phi  = alpha_eff;
if ~isfield(model,'pi3_hist') || isempty(model.pi3_hist), model.pi3_hist = zeros(0,4); end
model.pi3_hist(end+1,:) = [a_use, alpha_eff, double(pi_on)];   % [α_hg α_bnd α_eff gate](诊断, 断点续跑不保)
fprintf('[_PI-pil3] step %d : α_eff=%.3f gate=%d w_phi=%.3f | 退火 k=%d L=%.1f τ=%.2f\n', ...
    size(model.now_x,1), alpha_eff, pi_on, alpha_eff, knn_k, L_anneal, tau_drop);
occupied_table = mopso_adaptive_elitism_crowding_PI3(dim, bounds, window_size_bounds, ...
    model, num_samples, iterations1, iterations2, num_samples, elitism_size, ...
    model_num, roi_cache, fast_options);
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
