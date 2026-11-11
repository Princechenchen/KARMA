function [x_star, model] = mIMSE_refset(model)
% mIMSE_refset —— 对比方法 C3: 标准 IMSE(参考集离散化)
% 相对原始 sampling_Funciton/IMSEs.m 的改动(计划 §2.3 P0 修复):
%   1) 度量改为教科书 IMSE: x* = argmin_x ∫ tr Σ_{n+1}(z|x) dz,用统一参考集
%      (model.testsample,与其余臂共享)离散化。恒等关系: 加入 x 使 Σ_{n+1} 的迹
%      下降最多 ⟺ 参考集上总方差削减 Σ_z Σ_resp [σ²_n − σ²_{n+1}] 最大,
%      故直接 argmax 总削减。原始 IMSEs.m 的 min(sum(var)/(max−min)) 非标准。
%   2) 计算核与 mGlobal_SUR2w_varcs02lPD_fast_piBO 完全相同(含 3 处循环外预算),
%      但只用纯积分方差削减项 —— 不加 Cvar / cd / 距离项, 不乘任何先验。
%   3) 跨响应归一化规则显式声明: 各响应的方差削减除以该响应在参考集上的当前
%      最大预测方差 scale_j, 再跨响应求和(避免量纲支配)。
% 全局候选池、无 ROI、无先验,与统一协议一致。

ns = size(model.candidate,1); yd = size(model.init_value,2);
nts = size(model.testsample,1);

% 0) 参考集上当前预测方差 → 跨响应归一化尺度
[~, s_ref, model.MMRGP] = predict_resp(model.MMRGP, model.testsample);
scale = zeros(1,yd);
for i = 1:yd
    v = s_ref(i:yd:end, i);
    scale(i) = max(v);
    if scale(i) <= eps, scale(i) = 1; end
end
inv_scale = 1 ./ scale;

% 1) 候选点预测方差(扩充后协方差块要用)
[~, s, model.MMRGP] = predict_resp(model.MMRGP, model.candidate);

% 2) 当前模型常用量
zchol = poschol(model.MMRGP.z_mn);
invz  = zchol \ (zchol' \ eye(size(zchol)));
SigmaF = (zchol \ (zchol' \ model.MMRGP.fn));
FSigmaF = model.MMRGP.fn' * SigmaF;
FSigmaF_chol = poschol(FSigmaF);
invFSigmaF = FSigmaF_chol \ (FSigmaF_chol' \ eye(size(FSigmaF_chol)));
Im = eye(yd);

% ===== 循环外预算(与 _fast_piBO 同) =====
cov_TX = define_covmatrix0(model.MMRGP, model.testsample, model.MMRGP.X);
cov_cc = define_covmatrix0(model.MMRGP, model.candidate(1,:), model.candidate(1,:));
uFSigmaFu_all = zeros(yd, nts);
for jj = 1:nts
    cov_tX_tmp = cov_TX(1+yd*(jj-1):yd*jj, :);
    u_tmp = Im - cov_tX_tmp * SigmaF;
    uFSigmaFu_all(:, jj) = sum(u_tmp * invFSigmaF .* u_tmp, 2);
end

metric = zeros(ns, 1);
dvar = zeros(yd, nts);
for i = 1:ns
    candx = model.candidate(i,:);
    cov_ct = define_covmatrix0(model.MMRGP, candx, model.testsample);
    cov_cX = define_covmatrix0(model.MMRGP, candx, model.MMRGP.X);
    cov_cX_invz = cov_cX * invz;
    scandx = s(1+yd*(i-1):yd*i, :);
    invs = inv(scandx);

    F_new = [model.MMRGP.fn; Im];
    zmn_new = [[model.MMRGP.z_mn; cov_cX],[cov_cX'; cov_cc]];
    zchol_new = poschol(zmn_new);
    SigmaF_new = (zchol_new \ (zchol_new' \ F_new));
    FSigmaF_new = F_new' * SigmaF_new;
    FSigmaF_chol2 = poschol(FSigmaF_new);
    invFSigmaF_new = FSigmaF_chol2 \ (FSigmaF_chol2' \ eye(size(FSigmaF_chol2)));

    for j = 1:nts
        cov_tX = cov_TX(1+yd*(j-1):yd*j, :);          % 当前参考块
        COV = cov_ct(:, 1+yd*(j-1):yd*j) - cov_cX_invz * cov_tX';
        delta_sigma = sum(COV' * invs .* COV', 2);
        cov_tXcandx = [cov_tX, cov_ct(:, 1+yd*(j-1):yd*j)];
        u2 = Im - cov_tXcandx * SigmaF_new;
        uFSigmaFu_new = sum(u2 * invFSigmaF_new .* u2, 2);
        delta_beta = uFSigmaFu_all(:, j) - uFSigmaFu_new;
        dvar(:, j) = delta_sigma + delta_beta;
    end
    % 归一化后跨响应、跨参考点求和(总方差削减;argmax 等价于 argmin IMSE)
    metric(i) = sum(dvar .* inv_scale(:), 'all');
end

[~, index] = max(metric);
x_star = model.candidate(index, :);
model.M_ind(size(model.now_value,1) - size(model.init_value,1) + 1) = index;
model.imse_last_metric = metric;
end
