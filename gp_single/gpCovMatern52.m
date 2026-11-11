function [K, dK] = gpCovMatern52(logtheta, x, z, i)
% gpCovMatern52  Matern 5/2 核(逐维长度尺度 ARD)
%   超参 logtheta = [log l_1; ...; log l_d; log sf]
%   r2_ij = sum_k ((x_ik - z_jk)/l_k)^2,  r = sqrt(r2)
%   K_ij  = sf^2 * (1 + sqrt(5) r + 5 r^2/3) * exp(-sqrt(5) r)
%
%   K       = gpCovMatern52(lt, x)          自协方差(z 缺省取 x)
%   K       = gpCovMatern52(lt, x, z)       互协方差
%   [K, dK] = gpCovMatern52(lt, x, [], i)   dK/d logtheta_i

d   = size(x,2);
ell = exp(logtheta(1:d));  sf2 = exp(2*logtheta(d+1));
if nargin < 3 || isempty(z), z = x; end

D2 = zeros(size(x,1), size(z,1));
for k = 1:d
    D2 = D2 + ((x(:,k) - z(:,k)')./ell(k)).^2;
end
r  = sqrt(D2);
sq5 = sqrt(5);
K  = sf2 .* (1 + sq5*r + 5*D2/3) .* exp(-sq5*r);

if nargin == 4
    if i <= d
        Ei = ((x(:,i) - z(:,i)')./ell(i)).^2;       % dK/d log l_i = (5/3) sf2 e^{-sqrt5 r} (1+sqrt5 r) Ei
        dK = sf2 .* exp(-sq5*r) .* (5/3) .* (1 + sq5*r) .* Ei;
    else
        dK = 2*K;                                    % d/d log sf
    end
end
end
