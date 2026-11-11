function Mchol=poschol(M)
%POSCHOL 正定Cholesky分解（快速路径优化版）
%   先尝试直接chol分解，若矩阵不正定则fallback到eig+重构+chol
%   对于大部分正定矩阵，可避免昂贵的eig分解

M=abs(M);
M= triu(M)+triu(M)'-diag(diag(M));

% 快速路径：直接尝试chol分解
% 如果M已经是正定的（常见情况），直接返回，跳过eig
try
    Mchol=chol(M);
catch ME
    % Fallback：原方法（eig分解 + 截断负特征值 + 重构 + chol）
    [V,D]=eig(M);
    d=diag(D);
    d(d<=10e-10)=10e-10;
    Mtemp= V*diag(d)*V';
    Mchol=chol(Mtemp);
end
end
