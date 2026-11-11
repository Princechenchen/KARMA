function [x_star, model] =  MVAS(model)
ns = size(model.candidate,1); yd = size(model.init_value,2);
canddata = model.candidate; %原始候选集备份
[y_hat_candidate, s, model.MMRGP]=predict_resp(model.MMRGP,model.candidate);
for i=1:yd
    var(:,i) = s(i:yd:end,i);
end
% a. 找到候选点最近的已采样点（x*，原文定义）
distvar = dist(model.candidate,model.now_x'); %distance_weight
[~, idx_nearest] = min(distvar,[],2);
y_nearest = model.now_value(idx_nearest, :);  % 最近点的观测值
% b. 信息增益：||ŷ(x)-y(x*)||² + α×ΣMSPE_i（i=1..q）
term1 = sum((y_hat_candidate - y_nearest).^2,2);  % QoI预测误差
term2 = sum(var,2);  % QoI模型不确定性
term1 = (term1-min(term1))./(max(term1)-min(term1)); %归一化
term2 = (term2-min(term2))./(max(term2)-min(term2));
% 计算每个候选点的信息增益（无PQ，仅保留QoI项，原文公式11简化）
info_gain = term1 + term2;   
% 步骤4：计算旅行成本（Λ(x,xm)=||x-xm||/v，原文公式10）
travel_cost = sqrt(sum((model.candidate - model.now_x(end,:)).^2, 2));
travel_cost(travel_cost < 1e-3) = 1e-3;  % 避免除以0（当前点已采样）

% 步骤5：选择“信息增益/旅行成本”最大的点
metric = info_gain ./ travel_cost;
[bestvalue,index] = max(metric);
x_star = model.candidate(index,:);
[index] = ismember(canddata, x_star, 'rows');  %返回最佳点在原始候选集中的索引
index = find(index);
[index0] = ismember(model.candidate0, x_star, 'rows');  %返回最佳点在原始候选集中的索引
index0 = find(index0);
model.M_ind(size(model.now_value,1)-size(model.init_value,1)+1)=index0(1);
canddata(index,:)=[];
model.candidate = canddata;
% model.M_ind(size(model.now_value,1)-size(model.init_value,1)+1)=index;  %ship案例需要这个，且需要注释上面的7行%%
