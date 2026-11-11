function [x_star, model] = mSUR_ROI2wvarc_unified(model)
% mSUR_ROI2wvarc_unified —— 对比方法 C4: PAL-ROI-SUR(Python 分区 ROI + SUR 方差削减)
% 原始实现 sampling_Funciton/mSUR_ROI2wvarc.m 的统一协议修正版(原文件保持只读),
% 修正清单(见 对比方法与实验设计计划.md §2.4 P0):
%   1) 分类器随机种子固定: seed=int32(0)(可用 model.pal_seed 覆盖),可复现;
%      原版 seed=py.None → 逐次运行结果不可复现。
%   2) 拟合失败处理: 不再继承其它响应的分类器(原版语义 bug),改为同种子重建
%      分类器重试一次,仍失败则该响应退化为"全域 label 0"并继续。
%   3) 空 ROI 回退: 候选/参考子集筛空后回退全局池,不再崩溃或选出空点。
%   4) 去掉逐候选 fprintf 刷屏(I/O 开销)。
%   5) 计算核 3 处循环外预算(cov_TX / cov_cc / uFSigmaFu_all / cov_cX_invz),
%      与 _fast 系采样器相同的优化,数值结果不变。
% 其余机制(分区标签 → 方差最大标签 → yd-1 交集 ROI、SUR 方差削减、
% Cvar/cd/距离加和、选中点移出候选池)与原版一致。
% 依赖: ROImethod/partitioned_gp.py(需可用的 Python 环境 + scikit-learn)。

yd = size(model.init_value,2);
canddata  = model.candidate;     % 全局候选池备份
testsample = model.testsample;   % 全局参考集备份

%% ===== 1. Python 分区分类器识别各响应 ROI =====
py_path = py.sys.path;
[func_dir, ~, ~] = fileparts(mfilename('fullpath'));
parent_dir = fileparts(func_dir);
folder_path = fullfile(parent_dir, 'ROImethod');
insert(py_path, int32(0), folder_path);
pgp_module = py.importlib.import_module('partitioned_gp');

pal_seed = int32(0);
if isfield(model,'pal_seed') && ~isempty(model.pal_seed)
    pal_seed = int32(model.pal_seed);
end

X_init_py = py.numpy.array(model.now_x);
candx_py  = py.numpy.array(model.candidate);
testx_py  = py.numpy.array(model.testsample);

n_cand = size(model.candidate,1);
n_test = size(model.testsample,1);
CR_cand = zeros(yd, n_cand);
CR_test = zeros(yd, n_test);
for i = 1:yd
    Z_init_py = py.numpy.array(model.now_value(:,i));
    ok = true;
    ROIclf = pgp_module.pal_region_classifier(n_partitions=2, seed=pal_seed, n_iter=int32(300));
    try
        ROIclf.fit(X_init_py, Z_init_py);
    catch
        try   % 同种子重建重试一次(不继承其它响应的分类器)
            ROIclf = pgp_module.pal_region_classifier(n_partitions=2, seed=pal_seed, n_iter=int32(300));
            ROIclf.fit(X_init_py, Z_init_py);
        catch
            ok = false;
            fprintf('[C4-PAL] n=%d 响应%d 分区拟合失败,该响应退化为全域 label 0\n', ...
                size(model.now_x,1), i);
        end
    end
    if ok
        CR_cand(i,:) = double(ROIclf.predict(candx_py));
        CR_test(i,:) = double(ROIclf.predict(testx_py));
    else
        CR_cand(i,:) = 0; CR_test(i,:) = 0;
    end
end

%% ===== 2. 方差最大标签 → yd-1 交集 ROI(与原版同规则) =====
test_labels = unique(CR_test); test_labels = test_labels(:)';   % 候选/参考标签集可能不同,
                                                                % 比较一律用标签值本身

[~, s, model.MMRGP] = predict_resp(model.MMRGP, model.testsample);
var = zeros(n_test, yd);
for i = 1:yd
    var(:,i) = s(i:yd:end, i);
end
var = mapminmax(var', 0, 1); var = var';
sumvar = sum(var, 2);

indval = zeros(max([test_labels, 1]) + 1, yd);
for c = test_labels
    for i = 1:yd
        ind = CR_test(i,:) == c;
        indval(c+1, i) = sum(sumvar(ind));
    end
end
% 逐列 argmax(矩阵 max 默认按列): 每个响应取自己的方差最大标签,与原版一致;
% 随后 2/3 软交集(sum>=yd-1)。标签用值本身比较(避免两标签集排序不一致)。
% 注意 CR_cand/CR_test 是 yd×n,必须转置成 n×yd 再与 1×yd 标签行广播比较。
[~, ROIid] = max(indval);
roi_labels = test_labels(ROIid);     % 1×yd,各响应的 ROI 标签值

ind = CR_cand' == roi_labels;        % n_cand×yd,列 i 比较响应 i 的标签
candidateX = model.candidate(sum(ind) >= yd-1, :);
ind = CR_test' == roi_labels;
RefX = model.testsample(sum(ind) >= yd-1, :);

% 空 ROI 回退: 交集筛空则退回全局池(记录诊断)
model.c4_roi_fallback = false;
if size(candidateX,1) < 1 || size(RefX,1) < 1
    model.c4_roi_fallback = true;
    candidateX = canddata;
    RefX = testsample;
end
model.candidate  = candidateX;
model.testsample = RefX;
ns = size(model.candidate,1); nts = size(model.testsample,1);

%% ===== 3. SUR 方差削减核(_fast 同款预计算) =====
[~, s, model.MMRGP] = predict_resp(model.MMRGP, model.candidate);
var = zeros(ns, yd);
for i = 1:yd
    var(:,i) = s(i:yd:end, i);
end
zchol = poschol(model.MMRGP.z_mn);
invz  = zchol \ (zchol' \ eye(size(zchol)));
SigmaF = (zchol \ (zchol' \ model.MMRGP.fn));
FSigmaF = model.MMRGP.fn' * SigmaF;
FSigmaF_chol = poschol(FSigmaF);
invFSigmaF = FSigmaF_chol \ (FSigmaF_chol' \ eye(size(FSigmaF_chol)));
Im = eye(yd);

distvar = dist(model.candidate, model.now_x');
distvar = min(distvar, [], 2);

% 循环外预算
cov_TX = define_covmatrix0(model.MMRGP, model.testsample, model.MMRGP.X);
cov_cc = define_covmatrix0(model.MMRGP, model.candidate(1,:), model.candidate(1,:));
uFSigmaFu_all = zeros(yd, nts);
for jj = 1:nts
    cov_tX_tmp = cov_TX(1+yd*(jj-1):yd*jj, :);
    u_tmp = Im - cov_tX_tmp * SigmaF;
    uFSigmaFu_all(:, jj) = sum(u_tmp * invFSigmaF .* u_tmp, 2);
end

% 注意: metric 必须是 1×ns 行向量(原版未预分配,metric(i)=标量 从 1×1 沿行生长
% 成行向量)。若预分配成 ns×1 列,第 4 节 metric+Cvar'+cd'+distvar' 会广播成
% ns×ns 矩阵,max 返回 1×ns 下标向量 → x_star 变 ns×2。此处显式按行预分配。
metric = zeros(1, ns);
dvar = zeros(yd, nts);
for i = 1:ns
    candx = model.candidate(i,:);
    cov_ct = define_covmatrix0(model.MMRGP, candx, model.testsample);
    cov_cX = define_covmatrix0(model.MMRGP, candx, model.MMRGP.X);
    cov_cX_invz = cov_cX * invz;
    scandx = s(1+yd*(i-1):yd*i, :);
    invs = inv(scandx);

    F_new = [model.MMRGP.fn; Im];
    invz_new = blockm_inv(invz, cov_cX, cov_cc);   % 分块求逆平替增广重构
    SigmaF_new = invz_new * F_new;
    FSigmaF_new = F_new' * SigmaF_new;
    FSigmaF_chol2 = poschol(FSigmaF_new);
    invFSigmaF_new = FSigmaF_chol2 \ (FSigmaF_chol2' \ eye(size(FSigmaF_chol2)));

    for j = 1:nts
        cov_tX = cov_TX(1+yd*(j-1):yd*j, :);
        COV = cov_ct(:, 1+yd*(j-1):yd*j) - cov_cX_invz * cov_tX';
        delta_sigma = sum(COV' * invs .* COV', 2);
        cov_tXcandx = [cov_tX, cov_ct(:, 1+yd*(j-1):yd*j)];
        u2 = Im - cov_tXcandx * SigmaF_new;
        uFSigmaFu_new = sum(u2 * invFSigmaF_new .* u2, 2);
        delta_beta = uFSigmaFu_all(:, j) - uFSigmaFu_new;
        dvar(:, j) = delta_sigma + delta_beta;
    end
    meandvar = mean(dvar, 2); maxdvar = max(dvar, [], 2);
    weight = 0.5*(meandvar./sum(meandvar) + maxdvar./sum(maxdvar));
    metric(i) = sum(sum(dvar .* weight));
end

%% ===== 4. Cvar / cd / 距离加和(与原版同) =====
meanvar = mean(var); maxvar = max(var);
weight = 0.5*(meanvar./sum(meanvar) + maxvar./sum(maxvar));
Cvar = sum(var .* weight, 2);
cd = 1./exp(abs(model.MMRGP.fc));
cd = cd./max(cd); cd = mean(cd, 2);
Cvar = Cvar./max(Cvar);
metric = metric./max(metric);
metric = (metric + Cvar' + cd' + distvar');

%% ===== 5. 选中 + 池维护(选中点移出候选池,与原版同) =====
[~, index] = max(metric);
x_star = model.candidate(index, :);

[hit] = ismember(canddata, x_star, 'rows');
hit = find(hit);
[index0] = ismember(model.candidate0, x_star, 'rows');
index0 = find(index0);
model.M_ind(size(model.now_value,1) - size(model.init_value,1) + 1) = index0;
canddata(hit(1), :) = [];
model.candidate = canddata;
model.testsample = testsample;   % 还原全局参考集
model.last_roi_table = table();  % 无项目格式 ROI 表,占位空表供驱动器统一读取
end
