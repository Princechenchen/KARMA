function [x_star, model] = mGlobal_SUR2w_varcs02lPD_fast_QBC(model)
% mGlobal_SUR2w_varcs02lPD_fast_QBC - QBC-global 臂: 教科书 pool-based 查询委员会(2026-10-07)
% ICDE Task1 主动学习基线(论文口径)。与 mGlobal_SUR2w_varcs02lPD_fast_piBO(C6-global)
% 同款全局池约定: 【不跑 MOPSO/gen_ROIs】, 委员会分歧直接在档案供给的静态候选池
% (candidate0)上逐点评分、argmax 选点 —— 基线身份是"标准 AL", 不背 ROI 脚手架;
% 同脚手架换准则的消融位由 C4 承担, 本文件的 ROI 版(mROI_..._QBC.m)保留作 QBC-ROI
% 变体(与 C7-piBO-ROI 对称, 备审稿人索要)。
%
% 委员会准则(与 QBC-ROI 逐字一致):
%   K 个 MMRGP 在当前训练集的【无放回类别分层子采样】(比例 frac, 类内分别抽样)
%   上各自重拟合 —— 不用有放回 bootstrap(重复点使 corrgauss 相关阵奇异)。
%   分歧度 = 分类软分歧 clsDis(各委员 probit 类后验 p_k 的 Jensen-Shannon 散度:
%            clsDis = H(mean_k p_k) − mean_k H(p_k); ⚠QBC-ROI 冒烟实证: 硬投票熵
%            在 n≈30 时全体一致恒 0, 软分歧才能承载类通道信号)
%           + 连续响应跨委员会 std sd_j
%   score(x) = z(clsDis) + z(sd_y1) + z(sd_y2)   (逐池标准化, 等权, 无自由参数)
%
% 与 mGlobal_..._piBO 的约定对齐:
%   1) last_roi_table = table()(无区域表); M_ind 恒 0(非 ROI 驱动);
%   2) 静态池无重生成 —— 已标注点的委员会 std 由留出该点的少数委员贡献, 显著低于
%      未采区域, 重复选点风险低但仍可能(池协议固有属性, 与 C6-global 同担);
%   3) 每步首个 rng 消费者是委员会 randperm(无 MOPSO), 逐种子确定性。
% 诊断记录: qbc_terms/qbc_pick_index/qbc_step/qbc_hist(6列, 假定 2 连续响应)/
%   qbc_K_hist。可选字段 qbc_K(默认 8)、qbc_frac(默认 0.8)。无需先验字段。

ns = size(model.candidate,1); yd = size(model.init_value,2); nts = size(model.testsample,1);
canddata = model.candidate;
testsample = model.testsample;
%% 全局池口径(对齐 mGlobal_..._piBO): 不做任何区域识别, 池保持供给原样
model.last_roi_table = table();
model.candidate = canddata;
model.testsample = testsample;
ns = size(model.candidate,1); nts = size(model.testsample,1);
%% QBC 委员会(与 QBC-ROI 逐字一致)
ccols = find(model.ClassPos==1);          % 连续响应列(Branin ClassPos=[2,1,1] → [2 3])
cls_col = find(model.ClassPos==2);        % 训练值中的类别列
K = 8;   if isfield(model,'qbc_K')   && ~isempty(model.qbc_K),   K = model.qbc_K;   end
frac = 0.8; if isfield(model,'qbc_frac') && ~isempty(model.qbc_frac), frac = model.qbc_frac; end
n_train = size(model.now_x,1);
y_cls = model.now_value(:,cls_col);
idx1 = find(y_cls==1); idx0 = find(y_cls~=1);
m1 = max(round(frac*numel(idx1)), 1); m0 = max(round(frac*numel(idx0)), 1);
per1 = idx1(randperm(numel(idx1)));       % 类内无放回洗牌(分层子采样, 保类比)
per0 = idx0(randperm(numel(idx0)));
pred_c = zeros(ns, numel(ccols), K);      % 各委员连续预测
P_cls  = zeros(ns, K);                    % 各委员 probit 类后验 p_k
K_surv = 0; n_fail = 0;
t_com = tic;
for k = 1:K
    sub = [per1(1:min(m1,numel(per1))); per0(1:min(m0,numel(per0)))];
    try
        mk = MMRGP0(model.now_value(sub,:), model.now_x(sub,:), model.ClassPos, model.dim);
        [yk, ~, mk] = predict_resp(mk, model.candidate);  % 第3输出=挂 fc 的模型副本
        pk = (mk.fc + 1)/2;                              % 阈前 fc=2p-1 (probit 类后验)
        K_surv = K_surv + 1;
        P_cls(:,K_surv)  = min(max(pk(:), 1e-6), 1-1e-6);
        pred_c(:,:,K_surv) = yk(:,ccols);
    catch err_m
        n_fail = n_fail + 1;
        fprintf('[QBCg] 委员 %d 拟合失败(弃用): %s\n', k, err_m.message);
    end
end
t_com = toc(t_com);
if K_surv < 2, error('QBC:K_SURV', '存活委员 %d < 2, 委员会不可用', K_surv); end
P_cls = P_cls(:,1:K_surv);  pred_c = pred_c(:,:,1:K_surv);
% 分类软分歧: JS 散度 = 总类熵 − 委员平均类熵(连续取值; 硬投票熵在此数据量下恒 0)
Hfun = @(p) -(p.*log(p) + (1-p).*log(1-p));
clsDis = Hfun(mean(P_cls,2)) - mean(Hfun(P_cls), 2);
% 连续响应跨委员会 std
sd_mat = zeros(ns, numel(ccols));
for j = 1:numel(ccols)
    sd_mat(:,j) = std(pred_c(:,:,j), 0, 2);
end
zscore_safe = @(t) (t(:)-mean(t(:)))/max(std(t(:)),1e-12);   % 常数向量→全 0
zCD  = zscore_safe(clsDis);
zsd  = zeros(ns, numel(ccols));
for j = 1:numel(ccols)
    zsd(:,j) = zscore_safe(sd_mat(:,j));
end
score = zCD + sum(zsd, 2);
%% 教科书 pool 协议: 屏蔽已查询的池行【位置实例】(v2, 2026-10-11 修正)
% v1 按值屏蔽在 ship 上失效实录: INPUT 全表仅 45 唯一行, 池为有放回抽样(100 行仅
% ~25-30 唯一值), 按值屏蔽使每采一个新配置即屏蔽其全部重复池行 → 50 步预算内
% 必然池耗尽(ship 首跑 10 格全部 25-30 步 POOL_EXHAUSTED)。
% v2 语义 = 屏蔽已选的池位置(model.qbc_used_idx 追加记录): 刚选位置必屏蔽(防 bagging
%   留出伪影两连选), 同值其他实例行仍可选 = 对同一配置的重复测量(y 由行索引取该实例,
%   与 C8 定稿轮的 ismember 取首个匹配同属数据歧义范围内的既定口径)。对无重复池
%   (Branin/Ackley LHS 池)位置集合≡值集合, 与 v1 数学等价 → 已跑数据不受影响。
if ~isfield(model, 'qbc_used_idx') || isempty(model.qbc_used_idx)
    model.qbc_used_idx = zeros(1, 0);
end
labeled = false(ns, 1);
labeled(model.qbc_used_idx) = true;
n_masked = numel(model.qbc_used_idx);
if n_masked >= ns, error('QBC:POOL_EXHAUSTED', '候选池位置已全部查询'); end
model.qbc_labeled = labeled;           % 埋点: 离线重融合须施加同一屏蔽(位置级)
score(labeled) = -Inf;
[~, index] = max(score);
x_star = model.candidate(index,:);
model.qbc_used_idx = [model.qbc_used_idx, index];   % 记录本次查询位置(随 ma 返回)
%% 诊断记录(权重敏感性重融合 + 双通道相位 + 委员存活)
step_now = size(model.now_value,1) - size(model.init_value,1) + 1;
model.qbc_terms  = [zCD, zsd];            % [ns×(1+nRC)]
model.qbc_pick_index = index;
model.qbc_step = step_now;
c1 = 0; c2 = 0;
if std(zCD)>1e-12 && std(zsd(:,1))>1e-12, c1 = corr(zCD, zsd(:,1)); end
if size(zsd,2)>1 && std(zCD)>1e-12 && std(zsd(:,2))>1e-12, c2 = corr(zCD, zsd(:,2)); end
if ~isfield(model,'qbc_hist') || isempty(model.qbc_hist), model.qbc_hist = zeros(0,6); end
model.qbc_hist(end+1,:) = [step_now, c1, c2, mean(clsDis), max(sum(zsd,2)), index];
if ~isfield(model,'qbc_K_hist') || isempty(model.qbc_K_hist), model.qbc_K_hist = zeros(0,3); end
model.qbc_K_hist(end+1,:) = [step_now, K_surv, n_fail];
fprintf('[QBCg] step %d : K=%d/%d(败%d) 委员会 %.1f s | pool=%d(标%d) corr(zCD,zsd)=[%.3f %.3f] clsDis_mean=%.2e pick=(%.3f,%.3f) idx=%d\n', ...
    step_now, K_surv, K, n_fail, t_com, ns, n_masked, c1, c2, mean(clsDis), x_star(1), x_star(min(2,end)), index);
%% M_ind: global 臂恒 0(mGlobal_..._piBO 先例)
model.M_ind(size(model.now_value,1)-size(model.init_value,1)+1)=0;
model.candidate = canddata;
model.testsample = testsample;
end
