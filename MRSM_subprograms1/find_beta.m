function beta=find_beta(model, zchol)
%FIND_BETA 求解趋势函数系数beta（支持预计算Cholesky复用）
%   输入: model  - 包含 z_mn, ymn, fn 的模型结构体
%        zchol  - (可选) 已计算的 model.z_mn 的Cholesky分解
%                 若提供，则跳过内部的poschol调用，避免重复分解

if nargin < 2 || isempty(zchol)
    zchol=poschol(model.z_mn);
end

alpha1 = (zchol\(zchol'\model.ymn));
alpha2=(zchol\(zchol'\model.fn));
alpha3=model.fn'*alpha2;
alpha3chol=chol(alpha3);
alpha4=(alpha3chol\(alpha3chol'\model.fn'));
beta=alpha4*alpha1;

end
