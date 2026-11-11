function [x_star, model] = mROI_SUR2w_varcs02lPD_fast_PI_pil3br_DS_uconf(model)
% mROI_SUR2w_varcs02lPD_fast_PI_pil3br_DS_uconf —— pil3br_DS 的 uconf 裁判变体
% (2026-08-31, ship 第 5 轮先验攻坚: v3 几何 + box3 三件套 + uconf 裁判)。
% 与 pil3br_DS 逐行相同(池内选点+索引回映的数据集适配; box3 [C]先验场退火/
% [A]目标启停/[B]兑现加权; MOPSO 用 _PI3; L2 乘法禁用), 唯一差异(单因素受控):
%   分量裁判 loo_consistency_uniform_comp → loo_consistency_uniform_comp_uconf:
%   cons_hg 不变(连续 LOO 残差场), cons_bnd 换 corr(π_bnd, uconf),
%   uconf = 4·p̂(1−p̂)(probit 后验分类不确定性场)。
%   动机: ship v1/v3 几何 × 有无 box3 四格实验全部 α 双塌地板(V1 裁判不认任何
%   先验几何, [A]保险丝 47-48/50 步关闭先验列), 瓶颈定位在裁判而非几何;
%   Ackley 定稿(uconf + 4^dim 网格)已验证此武器, 本变体首次移植 ship;
%   v1box3 轮"手推符号带 cons_bnd 在线为正"亦提示分类通道问"模型还分不清处"更对症。
%   原 pil3br_DS 及更早全部变体零改动保留。
%
%   依赖 model 字段: pi_comp_fns={π_hg,π_bnd}, pi_comp_adaptive, pi_comp_eta,
%     pi_comp_alpha_min, pi_comp_alpha, pi_unif_grid, 可选 pi3_anneal{k,L,tau_drop}
%   不改动代理模型(MMRGP/trend_fun/MRSM)。

ns = size(model.candidate,1); yd = size(model.init_value,2); nts = size(model.testsample,1);
canddata = model.candidate;
testsample = model.testsample;
%% 关键区域分析确定候选集(根据关键区域生成候选样本)
dim = model.dim;
lower_bounds=zeros(1,dim);
upper_bounds=ones(1,dim);
bounds = [lower_bounds; upper_bounds]';
num_samples = 50;
iterations1 = 200;
RoI_num=floor(dim*2);
iterations2 = round(iterations1/RoI_num);
elitism_size = 10;
model_num=1;
window_min=0.35;
window_max=0.85;
window_size_bounds = zeros(dim, 2);
for d = 1:dim
    window_size_bounds(d, :) = [(upper_bounds(d) - lower_bounds(d)) * window_min, ...
        (upper_bounds(d) - lower_bounds(d)) * window_max];
end
roi_cache = prepare_roi_predict_cache(model.MMRGP);
fast_options = struct('batch_prediction', true, 'batch_archive', true, ...
    'reuse_sampling_template', true, 'vectorized_roi_metrics', true);

% ===== 分量独立 α（EMA on 均匀参考集 分量 consistency，地板 0.1；uconf 裁判）=====
if isfield(model,'pi_comp_adaptive') && model.pi_comp_adaptive
    eta = 0.5;  if isfield(model,'pi_comp_eta')       && ~isempty(model.pi_comp_eta),       eta  = model.pi_comp_eta;       end
    amin = 0.1; if isfield(model,'pi_comp_alpha_min') && ~isempty(model.pi_comp_alpha_min), amin = model.pi_comp_alpha_min; end
    a_prev = [1 1]; if isfield(model,'pi_comp_alpha') && ~isempty(model.pi_comp_alpha), a_prev = model.pi_comp_alpha; end
    uopts = struct('n_grid', 51);
    if isfield(model,'pi_unif_grid') && ~isempty(model.pi_unif_grid), uopts.n_grid = model.pi_unif_grid; end
    [cons_vec, ~] = loo_consistency_uniform_comp_uconf(model, uopts);   % uconf 裁判; 分量独立（不动 rng/代理模型）
    a_new = max(amin, eta*a_prev + (1-eta)*cons_vec);
    active_mask = [1 1];
    if isfield(model,'pi_comp_active_mask') && ~isempty(model.pi_comp_active_mask)
        active_mask = logical(model.pi_comp_active_mask);
    end
    a_new(~active_mask)=0;
    model.pi_comp_alpha = a_new;
    if ~isfield(model,'pi_comp_cons_hist')  || isempty(model.pi_comp_cons_hist),  model.pi_comp_cons_hist  = zeros(0,2); end
    if ~isfield(model,'pi_comp_alpha_hist') || isempty(model.pi_comp_alpha_hist), model.pi_comp_alpha_hist = zeros(0,2); end
    model.pi_comp_cons_hist(end+1,:)  = cons_vec;
    model.pi_comp_alpha_hist(end+1,:) = a_new;
    fprintf('[_PI-pil3br-uconf] step %d : α_hg %.3f→%.3f (cons %.3f) | α_bnd %.3f→%.3f (cons_uconf %.3f)\n', ...
        size(model.now_x,1), a_prev(1), a_new(1), cons_vec(1), a_prev(2), a_new(2), cons_vec(2));
end

% ===== α 加权融合；单路消融时禁用通道不参与 ROI 识别 =====
a_use = [1 1]; if isfield(model,'pi_comp_alpha') && ~isempty(model.pi_comp_alpha), a_use = model.pi_comp_alpha; end
active_mask = [1 1];
if isfield(model,'pi_comp_active_mask') && ~isempty(model.pi_comp_active_mask)
    active_mask = logical(model.pi_comp_active_mask);
end
fn_hg = model.pi_comp_fns{1};  fn_bnd = model.pi_comp_fns{2};
if all(active_mask == [1 0])
    fused_fn=fn_hg;
    fused_fn = @(X) fused_fn(X).^a_use(1);
elseif all(active_mask == [0 1])
    fused_fn=fn_bnd;
    fused_fn = @(X) fused_fn(X).^a_use(2);
else
    fused_fn = @(X) max(fn_hg(X).^a_use(1), fn_bnd(X).^a_use(2));
end

% ===== [box3-C 主] 先验场退火: π_eff(x) = fused(x)·min(1, ρ̂(x)^(-1/L)) =====
knn_k = 5;  L_anneal = 3;  tau_drop = 0.25;
if isfield(model,'pi3_anneal')
    if isfield(model.pi3_anneal,'k'),        knn_k   = model.pi3_anneal.k;         end
    if isfield(model.pi3_anneal,'L'),        L_anneal= model.pi3_anneal.L;         end
    if isfield(model.pi3_anneal,'tau_drop'), tau_drop= model.pi3_anneal.tau_drop;  end
end
Xv = model.now_x;
pi_eff_fn = @(X) fused_fn(X) .* prior_anneal_factor(X, Xv, knn_k, L_anneal);

% ===== [box3-A 辅] 目标启停 + [box3-B 辅] 兑现权重 =====
alpha_eff = max(a_use);
pi_on = alpha_eff >= tau_drop;
model.pi_phys_fn = pi_eff_fn;
model.pi_alpha   = 1;
model.pi3_active = pi_on;
model.pi3_w_phi  = alpha_eff;
if ~isfield(model,'pi3_hist') || isempty(model.pi3_hist), model.pi3_hist = zeros(0,4); end
model.pi3_hist(end+1,:) = [a_use, alpha_eff, double(pi_on)];
fprintf('[_PI-pil3br-uconf] step %d : α_eff=%.3f gate=%d w_phi=%.3f | 退火 k=%d L=%.1f τ=%.2f\n', ...
    size(model.now_x,1), alpha_eff, pi_on, alpha_eff, knn_k, L_anneal, tau_drop);
occupied_table = mopso_adaptive_elitism_crowding_PI3(dim, bounds, window_size_bounds, ...
    model, num_samples, iterations1, iterations2, num_samples, elitism_size, ...
    model_num, roi_cache, fast_options);
model.last_roi_table = occupied_table;

% --- ROI 候选生成(数据集版: 池内选点, 索引回映; 内联自 gen_ROI_DS 修正表格切片) ---
occupied = table2array(occupied_table); occupied = occupied(:,1:2*dim);
pool = canddata;
in_any = false(size(pool,1),1);
for i = 1:size(occupied,1)
    upb = occupied(i,2:2:end); dob = occupied(i,1:2:end);
    in_any = in_any | all(pool<=upb & pool>=dob, 2);
end
in_idx = find(in_any); out_idx = find(~in_any);
globalnum = floor(size(pool,1)*0.1);
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
zchol=poschol(model.MMRGP.z_mn);
invz = zchol\(zchol'\eye(size(zchol)));
SigmaF=(zchol\(zchol'\model.MMRGP.fn));
FSigmaF=model.MMRGP.fn'*SigmaF;
FSigmaF_chol=poschol(FSigmaF);
invFSigmaF = FSigmaF_chol\(FSigmaF_chol'\eye(size(FSigmaF_chol)));
Im = eye(size(model.ClassPos,2));

distvar = dist(model.candidate,model.now_x');
distvar = min(distvar,[],2);

% ===== 循环外预算 =====
cov_TX=define_covmatrix0(model.MMRGP,model.testsample,model.MMRGP.X);
cov_cc=define_covmatrix0(model.MMRGP,model.candidate(1,:),model.candidate(1,:));
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
    cov_ct=define_covmatrix0(model.MMRGP,candx,model.testsample);
    cov_cX=define_covmatrix0(model.MMRGP,candx,model.MMRGP.X);
    cov_cX_invz = cov_cX*invz;
    scandx = s(1+yd*(i-1):yd*i,:);
    invs = inv(scandx);
    F_new = [model.MMRGP.fn;Im];
    zmn_new =  [[model.MMRGP.z_mn;cov_cX],[cov_cX';cov_cc]];
    zchol_new=poschol(zmn_new);
    SigmaF_new=(zchol_new\(zchol_new'\F_new));

    FSigmaF_new=F_new'*SigmaF_new;
    FSigmaF_chol2=poschol(FSigmaF_new);
    invFSigmaF_new = FSigmaF_chol2\(FSigmaF_chol2'\eye(size(FSigmaF_chol2)));
    for j=1:nts
        cov_tX = cov_TX(1+yd*(j-1):yd*j,:);
        COV = cov_ct(:,1+yd*(j-1):yd*j) - cov_cX_invz*cov_tX';
        delta_sigma = sum(COV'*invs.*COV',2);

        cov_tXcandx=[cov_tX,cov_ct(:,1+yd*(j-1):yd*j)];
        u2=Im-cov_tXcandx*SigmaF_new;
        uFSigmaFu_new=sum(u2*invFSigmaF_new.*u2,2);

        delta_beta = uFSigmaFu_all(:,j)-uFSigmaFu_new;
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

% ===== L2 禁用（pil1 语义：metric 保持纯 M_data，先验只进 L1-a 区域识别）=====

[bestvalue,index] = max(metric);
x_star = model.candidate(index,:);
model.M_ind(size(model.now_value,1)-size(model.init_value,1)+1)=idcand(index);  % 池内索引(数据集查表用)
model.candidate = canddata;
model.testsample = testsample;
end
