function [x_star, model] =  maxMSE_ROI(model)
ns = size(model.candidate,1); yd = size(model.init_value,2);
canddata = model.candidate; %原始候选集备份
%% 关键区域分析确定候选集(根据关键区域生成候选样本)
dim = model.dim;
lower_bounds=zeros(1,dim);   % 搜索空间的下界   三组合
upper_bounds=ones(1,dim);   % 搜索空间的上界
bounds = [lower_bounds; upper_bounds]';  % 输入范围
num_samples = 50;  % 每个区域随机抽取的样本数量 80
iterations1 = 200;  % 迭代次数 200
RoI_num=4;%关键区域数量
iterations2 = round(iterations1/RoI_num);  %重启周期
elitism_size = 10;  % 精英数量
model_num=1;
window_min=0.35;%窗口相对原维度的最小比例
window_max=0.85;%窗口相对原维度的最大比例
window_size_bounds = zeros(dim, 2);
for d = 1:dim
    window_size_bounds(d, :) = [(upper_bounds(d) - lower_bounds(d)) * window_min, ...
        (upper_bounds(d) - lower_bounds(d)) * window_max];
end
occupied_table = mopso_adaptive_elitism_crowding(dim, bounds, window_size_bounds, model, num_samples, iterations1,iterations2, num_samples, elitism_size,model_num);
% [model.candidate,idcand] = gen_ROI_DS(model,ns,occupied_table); %要记得改
model.candidate = gen_ROIs(model,ns,occupied_table);
ns = size(model.candidate,1);

% 原
[y, s, model.MMRGP]=predict_resp(model.MMRGP,model.candidate);
for i=1:yd
    var(:,i) = s(i:yd:end,i);
end
maxvar = max(var,[],2);
[bestvalue,index] = max(maxvar);
x_star = model.candidate(index,:);
model.M_ind(size(model.now_value,1)-size(model.init_value,1)+1)=index;
model.candidate = canddata;
% model.M_ind(size(model.now_value,1)-size(model.init_value,1)+1)=idcand(index);