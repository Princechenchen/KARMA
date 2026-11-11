function [K, dK] = gpCovSEiso(logtheta, x, z, i)
% gpCovSEiso  平方指数核(单一长度尺度,各向同性)
%   超参 logtheta = [log l; log sf]
%   K(x,z)_ij = sf^2 * exp( -0.5 * |x_i - z_j|^2 / l^2 )
%
%   K       = gpCovSEiso(lt, x)          自协方差(z 缺省取 x)
%   K       = gpCovSEiso(lt, x, z)       互协方差
%   [K, dK] = gpCovSEiso(lt, x, [], i)   dK/d logtheta_i

ell = exp(logtheta(1));  sf2 = exp(2*logtheta(2));
if nargin < 3 || isempty(z), z = x; end

d  = size(x,2);
D2 = zeros(size(x,1), size(z,1));
for k = 1:d
    D2 = D2 + ((x(:,k) - z(:,k)')./ell).^2;
end
K = sf2 * exp(-0.5*D2);

if nargin == 4
    if i == 1
        dK = K .* D2;      % d/d log l
    else
        dK = 2*K;          % d/d log sf
    end
end
end
