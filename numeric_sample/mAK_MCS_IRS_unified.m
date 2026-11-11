function [x_star, model] = mAK_MCS_IRS_unified(model)
% mAK_MCS_IRS_unified —— 对比方法 C5: AK-MCS/IRS 风格主动学习(Relax-BM+分区)
% 原始实现 sampling_Funciton/AK_MCS_IRS2.m 的统一协议修正版(原文件保持只读)。
% 签名改为 [x_star, model] = f(model):不再接收 obj_fct。
% 相对原版的四点改造(见 对比方法与实验设计计划.md §2.5 P0):
%   1) 零内部真值调用: 原版每步 2 次 obj_fct(选中点 + 中心点)。统一版:
%      a) IR/UR 动态提升(原版用选中点真值符号)延迟到下一步入口,改读驱动器
%         已计费的观测 now_value 末行 —— 等效且零额外计费;
%      b) 中心点标签只算一次(初始化时,用 MMRGP 预测,与原版初始化路径一致)并缓存。
%   2) 参数化: mu/sigma/d/IF 可用 model.ak_mu/ak_sigma/ak_d/ak_IF 覆盖,
%      默认 mu=0.5, sigma=0.166 → TR≈[0.002,0.998]^dim 即全域, d=1/4, IF=0.8。
%   3) U 学习函数改为论文形式 U=|mu_g|/sigma_g: 原版分母误用方差、分子用硬标签 fc,
%      统一版分子用潜变量预测 fmu,分母用分类响应预测标准差 sqrt(var)。
%   4) IR/UR 抽样改无放回(randperm),避免原版 randi 有放回产生的重复点。
% 其余机制(TR/SR 划分、符号翻转定 IR、IF 配额、RR 冗余排除、候选池收缩)与原版一致。

xd = size(model.candidate, 2);
yd = size(model.init_value, 2);
canddata = model.candidate;      % 全局池备份(空池回退用)
class_col = find(model.ClassPos == 2);

%% 0) 参数(可覆盖)
mu    = 0.5*ones(1, xd);
sigma = 0.166*ones(1, xd);
d     = 1/4;
IF    = 0.8;
if isfield(model,'ak_mu')    && ~isempty(model.ak_mu),    mu    = model.ak_mu;    end
if isfield(model,'ak_sigma') && ~isempty(model.ak_sigma), sigma = model.ak_sigma; end
if isfield(model,'ak_d')     && ~isempty(model.ak_d),     d     = model.ak_d;     end
if isfield(model,'ak_IF')    && ~isempty(model.ak_IF),    IF    = model.ak_IF;    end

%% 1) 延迟的 IR/UR 动态提升(上一步选中点,真值已在 now_value 末行,零计费)
if isfield(model,'ak_pending') && model.ak_pending
    X_prev = model.now_x(end, :);
    G_prev = model.now_value(end, class_col);      % 分类真值标签
    if ~isempty(model.UR_idx)
        in_UR_flag = false; new_sr = [];
        for sr = model.UR_idx
            sr_lb = model.SR_list{sr}(1,:); sr_ub = model.SR_list{sr}(2,:);
            if all(X_prev >= sr_lb - 1e-8) && all(X_prev <= sr_ub + 1e-8)
                in_UR_flag = true; new_sr = sr; break;
            end
        end
        if in_UR_flag && sign(G_prev) ~= model.ak_center_sign
            if ~ismember(new_sr, model.IR_idx)
                model.IR_idx = [model.IR_idx, new_sr];
                model.UR_idx = model.UR_idx(model.UR_idx ~= new_sr);
            end
        end
    end
    model.ak_pending = false;
end

%% 2) 初始化(仅第一步): TR/SR 划分 + 用预测标签定 IR/UR(与原版初始化路径一致)
if ~isfield(model,'SR_list') || isempty(model.SR_list)
    TR_lb = mu - 3*sigma;  TR_ub = mu + 3*sigma;
    SR_list = cell(1, 2^xd);
    for i = 1:2^xd
        code = dec2bin(i-1, xd) - '0';
        SR_lb = TR_lb; SR_ub = TR_ub;
        for j = 1:xd
            if code(j) == 0, SR_ub(j) = mu(j);
            else,            SR_lb(j) = mu(j); end
        end
        SR_list{i} = [SR_lb; SR_ub];
    end
    % 边界点/中心点标签一律用 MMRGP 预测硬标签的符号(与原版初始化路径一致),零真值调用
    yc_hat = predict_resp(model.MMRGP, mu);
    model.ak_center_sign = sign(yc_hat(class_col));
    bound_pts = []; bits = dec2bin(0:2^xd-1, xd) - '0';
    for j = 1:xd
        dim_vals = [mu(j)-3*sigma(j), mu(j)+3*sigma(j)];
        bound_pts(:, j) = dim_vals(1) + bits(:, j) .* (dim_vals(2) - dim_vals(1));
    end
    bound_pts = min(max(bound_pts, 0), 1);          % 裁剪到 [0,1] 域
    yb_hat = predict_resp(model.MMRGP, bound_pts);
    bound_sign = sign(yb_hat(:, class_col));
    IR_idx = []; UR_idx = [];
    for i = 1:size(bound_pts, 1)
        bound_pt = bound_pts(i, :);
        for sr = 1:2^xd
            sr_lb = SR_list{sr}(1,:); sr_ub = SR_list{sr}(2,:);
            in_sr = all(bound_pt >= sr_lb - 1e-8) && all(bound_pt <= sr_ub + 1e-8);
            if in_sr
                if model.ak_center_sign ~= bound_sign(i) && ~ismember(sr, IR_idx)
                    IR_idx = [IR_idx, sr];
                elseif ~ismember(sr, IR_idx) && ~ismember(sr, UR_idx)
                    UR_idx = [UR_idx, sr];
                end
                break;
            end
        end
    end
    model.SR_list = SR_list;
    model.IR_idx  = IR_idx;
    model.UR_idx  = UR_idx;
end

%% 3) IR/UR 样本池(从当前候选池筛,RR 排除已在上几步收缩过池)
X_TR = model.candidate;
ns = size(X_TR, 1);
X_IR = []; X_UR = [];
for sr = model.IR_idx
    sr_lb = model.SR_list{sr}(1,:); sr_ub = model.SR_list{sr}(2,:);
    in_box = all(X_TR >= sr_lb - 1e-8, 2) & all(X_TR <= sr_ub + 1e-8, 2);
    X_IR = [X_IR; X_TR(in_box, :)];
end
for sr = model.UR_idx
    sr_lb = model.SR_list{sr}(1,:); sr_ub = model.SR_list{sr}(2,:);
    in_box = all(X_TR >= sr_lb - 1e-8, 2) & all(X_TR <= sr_ub + 1e-8, 2);
    X_UR = [X_UR; X_TR(in_box, :)];
end
if isempty(X_IR) && isempty(X_UR)      % 池耗尽回退: 用全局池
    X_IR = canddata;
end

% IF 配额 + 无放回抽样(randperm)
N_IR = min(round(IF * ns), size(X_IR, 1));
N_UR = min(ns - round(IF * ns), size(X_UR, 1));
if isempty(X_IR)
    idx_UR = randperm(size(X_UR,1), N_UR);
    X_candidate = X_UR(idx_UR, :);
elseif isempty(X_UR)
    idx_IR = randperm(size(X_IR,1), N_IR);
    X_candidate = X_IR(idx_IR, :);
else
    idx_IR = randperm(size(X_IR,1), N_IR);
    idx_UR = randperm(size(X_UR,1), N_UR);
    X_candidate = [X_IR(idx_IR, :); X_UR(idx_UR, :)];
end

%% 4) U 学习函数(论文形式): U = |fmu| / sigma_c, 选 U 最小
[~, s, model.MMRGP] = predict_resp(model.MMRGP, X_candidate);
nc = size(X_candidate, 1);
var = zeros(nc, yd);
for i = 1:yd
    var(:, i) = s(i:yd:end, i);
end
fmu_c  = model.MMRGP.fmu(:, 1);                   % 分类潜变量预测
sigma_c = sqrt(max(var(:, class_col), 0));        % 分类响应预测标准差
U = abs(fmu_c) ./ max(sigma_c, eps);
[~, index] = min(U);
X_new = X_candidate(index, :);
x_star = X_new;

%% 5) RR 冗余排除(从候选池删除 X_new±d*sigma 盒内的点,与原版同)
RR_lb = X_new - d*sigma;  RR_ub = X_new + d*sigma;
in_RR = all(X_TR >= RR_lb - 1e-8, 2) & all(X_TR <= RR_ub + 1e-8, 2);
model.candidate = X_TR(~in_RR, :);

model.ak_pending = true;      % 下一步入口做延迟 IR/UR 提升
[index0] = ismember(model.candidate0, x_star, 'rows');
index0 = find(index0);
model.M_ind(size(model.now_value,1) - size(model.init_value,1) + 1) = index0;
model.last_roi_table = table();   % 无项目格式 ROI 表,占位供驱动器统一读取
end
