function [K, dK] = gpCovNoise(logtheta, x, z, i)
% gpCovNoise  独立高斯观测噪声核(仅自协方差有值)
%   超参 logtheta = [log sn](观测噪声标准差)
%   K(x,x) = sn^2 * I;   K(x,z) = 0 (x≠z)
%
%   K       = gpCovNoise(lt, x)          n×n 对角阵
%   [K, dK] = gpCovNoise(lt, x, [], 1)   dK/d log sn = 2K

if nargin < 3 || isempty(z), z = x; end
n  = size(x,1);  m = size(z,1);
sn2 = exp(2*logtheta(1));

if n == m && isequal(x, z)
    K = sn2*eye(n);
else
    K = zeros(n,m);
end

if nargin == 4
    dK = 2*K;
end
end
