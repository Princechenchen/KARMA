function [global_best_position,global_best_fitness] = PSO_optimize(hyp,y,x)
% 初始化参数
x0= [hyp.cov,hyp.lik];
xmax=5; xmin = 1E-3;
num_particles = 100; % 粒子数量
num_dimensions = length(x0); % 搜索空间维度
max_iterations = 500; % 最大迭代次数
w = 0.5; % 惯性权重
c1 = 2; % 个体学习因子
c2 = 2; % 全局学习因子
[~,dist_matrix]= covSEfun2(hyp,x,x);   %协方差矩阵
% lower_bound = -5; % 搜索空间下界
% upper_bound = 5; % 搜索空间上界
vlimit = xmax/10.*ones(size(x0));  %速度限制
% 初始化粒子位置和速度
particles_position=zeros(num_particles,num_dimensions);
% particles_fitness = zeros(num_particles,1);
particles_position = particles_position+(xmax-xmin).*rand(num_particles,num_dimensions);
particles_fitness = fitfunc(particles_position,y,dist_matrix);
particles_velocity = zeros(num_particles,num_dimensions);
% 初始化个体最优位置和全局最优位置
particles_best_position = particles_position;
particles_best_fitness = particles_fitness;
% global_best_position = zeros(num_n,num_dimensions);
% global_best_fitness = inf;
[global_best_fitness,ind] = max(particles_best_fitness) ;%30个试验点为例
global_best_position = particles_best_position(ind,:);
result = zeros(1,max_iterations);
% 开始迭代
for iteration = 1:max_iterations
    % 更新粒子速度和位置
    r1 = rand(num_particles,num_dimensions);
    r2 = rand(num_particles,num_dimensions);
    particles_velocity = w*particles_velocity+c1 * r1 .*(particles_best_position - particles_position) + c2 * r2 .* (global_best_position - particles_position);
    %调整速度限制
    for j=1:num_dimensions 
        index1 = find(particles_velocity(:,j)>vlimit(j));
        index2 = find(particles_velocity(:,j)<-1*vlimit(j));
        particles_velocity(index1,j)=vlimit(j);
        particles_velocity(index2,j)=-1*vlimit(j);
    end
    %更新位置
    particles_position = particles_position + particles_velocity; 
    %调整位置限制
    for j=1:num_dimensions 
        index1 = find(particles_position(:,j)>xmax);
        index2 = find(particles_position(:,j)<xmin);
        particles_position(index1,j)=xmax;
        particles_position(index2,j)=xmin;
    end
    % 计算每个粒子的适应度值
    particles_fitness = fitfunc(particles_position,y,dist_matrix);
    % 更新个体最优位置和全局最优位置
    ind_pd = find(particles_fitness> particles_best_fitness);
    particles_best_fitness(ind_pd,:) = particles_fitness(ind_pd,:);
    particles_best_position(ind_pd,:) = particles_position(ind_pd,:);
    
    [best_fitness,ind] = max(particles_fitness) ;
    if best_fitness>global_best_fitness
        global_best_fitness = best_fitness;
        global_best_position = particles_position(ind,:);
    end
    result(iteration) = global_best_fitness;
    
end
% plot([1:100],result)
% xlabel('迭代次数')
% ylabel('目标函数值')
% % 输出结果
% disp(['Global best position: ' num2str(global_best_position)]);
% disp(['Global best fitness: ' num2str(global_best_fitness)]);
%
% % 定义适应度函数
% function fitness = fitness_function(x)
%     % 这里需要根据具体问题定义适应度函数
% end
% 定义适应度函数
function likeval = fitfunc(x,y,dist_matrix)
%x-超参数 y-响应 dist_matrix-距离矩阵
n= numel(y);
for i=1:size(x,1)
K = x(i,2)^2.*exp(-dist_matrix/2/(x(i,1)^2));
L = chol(K+x(i,3).^2.*eye(n),'lower');  %cholesky分解，K求逆
alpha = L'\(L\y);
likeval(i,1) = -1/2*y'*alpha-sum(diag(log(L)))-n/2*log(2*pi); 
end
end
end