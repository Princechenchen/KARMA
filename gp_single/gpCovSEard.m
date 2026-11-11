function [K, dK] = gpCovSEard(logtheta, x, z, i)
% gpCovSEard  平方指数核(逐维长度尺度 ARD)
%   超参 logtheta = [log l_1; ...; log l_d; log sf](列向量,d = 输入维数)
%   K(x,z)_ij = sf^2 * exp( -0.5 * sum_k ((x_ik - z_jk)/l_k)^2 )
%
%   K           = gpCovSEard(lt, x)          自协方差 n×n(z 缺省取 x)
%   K           = gpCovSEard(lt, x, z)       互协方差 n×m
%   [K, dK]     = gpCovSEard(lt, x, [], i)   额外返回 dK/d logtheta_i(gp_nll 求梯度用)

d   = size(x,2);
ell = exp(logtheta(1:d));  sf2 = exp(2*logtheta(d+1));
if nargin < 3 || isempty(z), z = x; end

D2 = zeros(size(x,1), size(z,1));
for k = 1:d
    D2 = D2 + ((x(:,k) - z(:,k)')./ell(k)).^2;    % 隐式扩展 n×1 − 1×m
end
K = sf2 * exp(-0.5*D2);

if nargin == 4
    if i <= d
        dK = K .* (((x(:,i) - z(:,i)')./ell(i)).^2);  % d/d log l_i
    else
        dK = 2*K;                                     % d/d log sf
    end
end
end
