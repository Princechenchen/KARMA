function [x_star, model] =  mKMDT(model)
ns = size(model.candidate,1); yd = size(model.init_value,2);
% [y, s, model.MMRGP]=predict_resp(model.MMRGP,model.candidate);
% for i=1:yd
%     var(:,i) = s(i:yd:end,i);
% end
% maxvar = max(var,[],2);
% [bestvalue,index] = max(maxvar);
% x_star = model.candidate(index,:);

X = model.now_x; d = size(X,2);
if d<=3 &&d>1
    % -------------------- 步骤1：德劳内三角剖分设计空间 --------------------
    % 对当前样本点进行三角剖分（仅支持2D/3D，高维需降维）
    if d == 2
        tri = delaunay(X(:, 1), X(:, 2));  % 2D三角剖分，tri为三角形顶点索引（m×3矩阵，m为三角形数量）
    elseif d == 3
        tri = delaunay(X(:, 1), X(:, 2), X(:, 3));  % 3D四面体剖分（文档中三角剖分的高维扩展）
    else
        error('当前代码仅支持2D/3D，高维需结合PCA降维（文档1-203）');
    end
    m = size(tri, 1);  % 三角形数量
    % -------------------- 步骤2：计算全局探索指标（三角形面积） --------------------
    area = zeros(m, 1);
    for i = 1:m
        vertices = X(tri(i, :), :);  % 第i个三角形的3个顶点（d维）
        if d == 2
            % 2D三角形面积公式：1/2|(x2-x1)(y3-y1)-(x3-x1)(y2-y1)|
            area(i) = 0.5 * abs(det([vertices(2,:)-vertices(1,:); vertices(3,:)-vertices(1,:)]));
        elseif d == 3
            % 3D四面体体积（类比2D面积，反映空间分散度）
            area(i) = abs(det([vertices(2,:)-vertices(1,:); vertices(3,:)-vertices(1,:); vertices(4,:)-vertices(1,:)])) / 6;
        end
    end
    % 计算每个三角形形心的预测误差（MSE，文档1-31）
    centroid = zeros(m, d);  % 各三角形形心（d维）
    for i = 1:m
        vertices = X(tri(i, :), :);
        centroid(i, :) = mean(vertices, 1);  % 形心坐标（顶点平均）
    end
    %--------------局部---------------%
    % 计算每个输出在形心处的预测误差（MSE）
    err = zeros(m, yd);  % err(i,k)：第i个三角形形心在第k个输出的预测误差
    [~, s, ~]=predict_resp(model.MMRGP,centroid);
    for i=1:yd
        err(:,i) = s(i:yd:end,i);
    end
    sampx = centroid;
else %d>3 使用最小距离进行全局优化
    distvar = dist(model.candidate,model.now_x'); %distance_weight
    area = min(distvar,[],2);
    %--------------局部---------------%
    err = zeros(ns, yd);
    [~, s, ~]=predict_resp(model.MMRGP,model.candidate);
    for i=1:yd
        err(:,i) = s(i:yd:end,i);
    end
    sampx = model.candidate;
end

% -------------------- 步骤4：构建决策矩阵+熵权法求权重 ---------------------
% 4.1 构建初始决策矩阵A（m个方案，1个面积属性 + t个误差属性）
A = [area, err];  % m × (t+1)矩阵，列1：面积，列2~t+1：各输出误差
[m, n] = size(A);   % n为属性数（t+1个）

% 4.2 决策矩阵归一化（公式12：正向指标，越大越优）
B = zeros(m, n);
for j = 1:n
    B(:, j) = A(:, j) / sum(A(:, j));  % 归一化到[0,1]
end

% 4.3 熵权法计算属性权重（公式13-15）
e = zeros(1, n);  % 各属性熵值
k = 1 / log(m);   % 熵系数
for j = 1:n
    p = B(:, j) / sum(B(:, j));  % 概率分布（避免0值，加微小量）
    p(p == 0) = 1e-10;
    e(j) = -k * sum(p .* log(p));
end
w = (1 - e) / sum(1 - e);  % 权重（熵越小，权重越大）
% -------------------- 步骤5：TOPSIS选择敏感区域（文档1-69至1-73） --------------------
% 5.1 加权决策矩阵Z
Z = B .* w;

% 5.2 正理想解z1（各属性最大值）、负理想解z2（各属性最小值）
z1 = max(Z, [], 1);  % 正向指标：越大越优
z2 = min(Z, [], 1);

% 5.3 计算欧氏距离S1（到正理想解）、S2（到负理想解）（公式17-18）
S1 = sqrt(sum((Z - z1).^2, 2));
S2 = sqrt(sum((Z - z2).^2, 2));

% 5.4 计算贴近度T（公式19），T越大，区域越敏感
T = S2 ./ (S1 + S2);
[~, index] = max(T);  % 敏感三角形索引
% -------------------- 步骤6：添加新样本点（敏感三角形形心） --------------------
x_star = sampx(index, :);  % 新样本点（形心）
model.M_ind(size(model.now_value,1)-size(model.init_value,1)+1)=index;
