function [x_star, model] = mROI_SUR2w_varcs02lPD_fast_PI_comp_DS_b3u(model)
% mROI_SUR2w_varcs02lPD_fast_PI_comp_DS_b3u —— wingweight 先验臂采样器(2026-08-30)
% 对应臂名 C8uc-box3u-COMP: Branin 定稿 C8uc-box3-COMP 的先验处理方式在
% 10D 池协议下的移植(原 _comp_DS / _pil3 / _uconf 及更早变体全部零改动保留)。
%
% 组装关系(三份模板逐块拼合):
%   [骨架] _comp_DS: ROI 候选=池内选点(table 前 2*dim 列窗口, 池点在窗内
%     入选, 另保留 10% 全局池点防集中; model.M_ind 记池内索引, 供驱动器
%     校验)。testsample 不重新生成(保持固定参考集)。
%   [裁判] uconf 单臂(用户 2026-08-30 指示): loo_consistency_uniform_comp_uconf_sobol
%     —— cons_hg=corr(π_hg, 连续 LOO 残差场), cons_bnd=corr(π_bnd, uconf=4p̂(1−p̂));
%     dim≥8 且给 pi_unif_sobol_n 时参考集=Sobol 点集(sobolset Skip 4096/Leap 1,
%     rng 保护, 与 _comp.m 十维先例逐字相同; 51^10 网格维度诅咒不可行)。
%   [box3 三件套] _pil3(与 Branin C8uc-box3-COMP 完全一致):
%     [C 主] 先验场退火: pi_eff = fused·min(1, ρ̂^(−1/L))(prior_anneal_factor,
%            k=5, L=3; ρ̂=kNN 局部密度/平均密度) —— 超采区先验退休, 场随
%            now_x 每步演化, 可经 model.pi3_anneal 配置;
%     [A 辅] 目标启停: α_eff=max(α_hg,α_bnd) < τ_drop=0.25 → 当步 MOPSO 回
%            3 目标(model.pi3_active 传 PI3);
%     [B 辅] 兑现加权: commit top-k 质量分加权随机, Φ 列权重 w_phi=α_eff
%            (model.pi3_w_phi 传 PI3)。
%   [L1-only] fused=max(π_hg^α_hg, π_bnd^α_bnd) 只进 MOPSO 第 4 目标
%     (pi_phys_fn=退火场, pi_alpha=1 → Φ_eff=mean_R(pi_eff));
%     L2 点级乘法 S(x)=M_data·fused 禁用(box3/pil1 语义, 乘法行注释保留)。
%
% 分量 α: EMA η=0.5, 地板 0.1(同 _comp_DS)。不改动代理模型/gen_ROIs/历史采样器。
% 依赖 model 字段: pi_comp_fns={π_hg,π_bnd}(必需), pi_comp_adaptive, pi_comp_eta,
%   pi_comp_alpha_min, pi_comp_alpha, pi_unif_sobol_n(可选, dim≥8 生效),
%   pi3_anneal(可选, 默认 k=5/L=3/tau_drop=0.25)。

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

% ===== 分量独立 α（EMA on 均匀参考集 分量 consistency，地板 0.1）=====
if isfield(model,'pi_comp_adaptive') && model.pi_comp_adaptive
    eta = 0.5;  if isfield(model,'pi_comp_eta')       && ~isempty(model.pi_comp_eta),       eta  = model.pi_comp_eta;       end
    amin = 0.1; if isfield(model,'pi_comp_alpha_min') && ~isempty(model.pi_comp_alpha_min), amin = model.pi_comp_alpha_min; end
    a_prev = [1 1]; if isfield(model,'pi_comp_alpha') && ~isempty(model.pi_comp_alpha), a_prev = model.pi_comp_alpha; end
    uopts = struct('n_grid', 51);
    if isfield(model,'pi_unif_grid') && ~isempty(model.pi_unif_grid), uopts.n_grid = model.pi_unif_grid; end
    if isfield(model,'pi_unif_sobol_n') && ~isempty(model.pi_unif_sobol_n) ...
            && model.dim >= 8 && ~isempty(which('sobolset'))
        sob = sobolset(model.dim, 'Skip', 4096, 'Leap', 1);
        rng_sob_guard = rng;  rng(2026, 'twister');   % 参考集固定(种子无关性)
        uopts.ref_set = net(sob, model.pi_unif_sobol_n);
        rng(rng_sob_guard);
    end
    [cons_vec, ~] = loo_consistency_uniform_comp_uconf_sobol(model, uopts);   % 分量独立（不动 rng/代理模型）
    a_new = max(amin, eta*a_prev + (1-eta)*cons_vec);
    model.pi_comp_alpha = a_new;
    % 轨迹记录（护栏初始化）
    if ~isfield(model,'pi_comp_cons_hist')  || isempty(model.pi_comp_cons_hist),  model.pi_comp_cons_hist  = zeros(0,2); end
    if ~isfield(model,'pi_comp_alpha_hist') || isempty(model.pi_comp_alpha_hist), model.pi_comp_alpha_hist = zeros(0,2); end
    model.pi_comp_cons_hist(end+1,:)  = cons_vec;
    model.pi_comp_alpha_hist(end+1,:) = a_new;
    fprintf('[_PI-DSb3u] step %d : α_hg %.3f→%.3f (cons %.3f) | α_bnd %.3f→%.3f (cons %.3f)\n', ...
        size(model.now_x,1), a_prev(1), a_new(1), cons_vec(1), a_prev(2), a_new(2), cons_vec(2));
end

% ===== α 加权 max 融合（L1 与 L2(禁用) 共用）=====
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
fprintf('[_PI-DSb3u] step %d : α_eff=%.3f gate=%d w_phi=%.3f | 退火 k=%d L=%.1f τ=%.2f\n', ...
    size(model.now_x,1), alpha_eff, pi_on, alpha_eff, knn_k, L_anneal, tau_drop);
occupied_table = mopso_adaptive_elitism_crowding_PI3(dim, bounds, window_size_bounds, ...
    model, num_samples, iterations1, iterations2, num_samples, elitism_size, ...
    model_num, roi_cache, fast_options);
model.last_roi_table = occupied_table;   % 【诊断记录】暴露本次识别的关键区域集(含 Phi 目标)，供可视化/存档

% --- ROI 候选生成(数据集版: 池内选点, 索引回映; 内联自 gen_ROI_DS 修正表格切片) ---
occupied = table2array(occupied_table); occupied = occupied(:,1:2*dim);
pool = canddata;                                   % 当前候选池
in_any = false(size(pool,1),1);
for i = 1:size(occupied,1)
    upb = occupied(i,2:2:end); dob = occupied(i,1:2:end);
    in_any = in_any | all(pool<=upb & pool>=dob, 2);
end
in_idx = find(in_any); out_idx = find(~in_any);
globalnum = floor(size(pool,1)*0.1);               % 保留 10% 全局点防集中(同 gen_ROI_DS)
if numel(out_idx) < globalnum
    sel = [in_idx; out_idx];
else
    rp = randperm(numel(out_idx), globalnum);
    sel = [in_idx; out_idx(rp(:))];
end
idcand = sel;
model.candidate = pool(sel,:);
ns = size(model.candidate,1);
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
% [box3/L1-only] L2 乘法禁用(pil1 语义): metric stays pure M_data          % 对齐 [1 × ns]

[bestvalue,index] = max(metric);
x_star = model.candidate(index,:);
model.M_ind(size(model.now_value,1)-size(model.init_value,1)+1)=idcand(index);  % 池内索引(数据集查表用)
model.candidate = canddata;
model.testsample = testsample;
end
