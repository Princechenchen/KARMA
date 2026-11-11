%***********模型参数获取****************
% 包调用
disp ('executing gpml startup script...')
mydir='F:\学习工作\搞点研究\混合响应建模\单一响应建模\';        % where am I located
% mydir='F:\学习工作-wzc\搞点研究\混合响应建模\单一响应建模'; 
addpath (mydir);
% core folders
dirs = {'cov','doc','inf','lik','mean','prior','util'};
for d = dirs
  addpath (fullfile (mydir, d{1}))
end
% minfunc folders
dirs = {{'util','minfunc'},{'util','minfunc','compiled'}};
for d = dirs
  addpath (fullfile (mydir, d{1}{:}))
end
addpath([mydir,'/util/sparseinv'])

% 读取数据
addpath('F:\学习工作\搞点研究\混合响应序贯采样\新建文件夹\sampling\MRSM_subprograms');  
addpath('F:\学习工作\搞点研究\混合响应序贯采样\新建文件夹\sampling');
load rec1017_SUR_betaU_ROI.mat
x = model.init_x;
y = model.init_value;


% **********************入参*************
dim = model.dim;  % 输入因子维度
lower_bounds=zeros(1,dim);   % 搜索空间的下界   三组合
upper_bounds=ones(1,dim);   % 搜索空间的上界
num_samples = 80;  % 每个区域随机抽取的样本数量
iterations1 = 200;  % 迭代次数
RoI_num=4;%关键区域数量
iterations2 = round(iterations1/RoI_num);  %重启周期
elitism_size = 10;  % 精英数量
window_min=0.35;%窗口相对原维度的最小比例
window_max=0.85;%窗口相对原维度的最大比例
location='F:\学习工作\搞点研究\混合响应序贯采样\新建文件夹\sampling\baseline\MS-PSO\ROI_Results.xlsx';
model_num=1;
% model=1;
% ************end*********************
objective_func = @(A) Y(A, model, model_num);
bounds = [lower_bounds; upper_bounds]';  % 输入范围
window_size_bounds = zeros(dim, 2);
for d = 1:dim
    window_size_bounds(d, :) = [(upper_bounds(d) - lower_bounds(d)) * window_min, ...
        (upper_bounds(d) - lower_bounds(d)) * window_max];
end
reference_samples=latin_hypercube_sampling(bounds,10000);
kde_q_ = @(x) mvksdensity(reference_samples, x, 'Bandwidth', 0.2);
% all_extrema = find_local_extrema(objective_func, dim, 50,lower_bounds, upper_bounds, 'BFGS', 1e-5);
% ************MSPSO算法执行*********************
occupied_table = mopso_adaptive_elitism_crowding(dim, bounds, window_size_bounds, model, num_samples, iterations1,iterations2, num_samples, elitism_size,model_num);

