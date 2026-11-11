function [global_best_position,global_best_fitness,alpha_best] = PSO_optimize_c(hyp,y,x)
% 初始化参数
x0= hyp;
% xmax=[10,5,5,5,5]; xmin = [-10,1E-3,1E-3,1E-3,1E-3];  %1E-3
xmax=10; xmin=-10;
num_particles = 50; % 粒子数量
num_dimensions = length(x0); % 搜索空间维度
max_iterations = 100; % 最大迭代次数
w = 0.5; % 惯性权重
c1 = 2; % 个体学习因子
c2 = 2; % 全局学习因子
% [~,dist_matrix]= covSEfun2(hyp,x,x);   %协方差矩阵
% lower_bound = -5; % 搜索空间下界
% upper_bound = 5; % 搜索空间上界
vlimit = xmax./10.*ones(size(x0));  %速度限制
alpha =  zeros(size(y,1),num_particles);  %每一列对应一个粒子
% 初始化粒子位置和速度
particles_position=zeros(num_particles,num_dimensions);
% particles_fitness = zeros(num_particles,1);
particles_position = particles_position+(xmax-xmin).*rand(num_particles,num_dimensions);
[particles_fitness,alpha] = fitfun_class(particles_position,y,x,alpha);
particles_velocity = zeros(num_particles,num_dimensions);
% 初始化个体最优位置和全局最优位置
particles_best_position = particles_position;
particles_best_fitness = particles_fitness;
% global_best_position = zeros(num_n,num_dimensions);
% global_best_fitness = inf;
[global_best_fitness,ind] = max(particles_best_fitness);%30个试验点为例
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
    [particles_fitness,alpha] = fitfun_class(particles_position,y,x,alpha);
    
    % 更新个体最优位置和全局最优位置
    ind_pd = find(particles_fitness> particles_best_fitness);
    particles_best_fitness(ind_pd,:) = particles_fitness(ind_pd,:);
    particles_best_position(ind_pd,:) = particles_position(ind_pd,:);
    
    [best_fitness,ind] = max(particles_fitness) ;
    if best_fitness>global_best_fitness
        global_best_fitness = best_fitness;
        global_best_position = particles_position(ind,:);
        alpha_best = alpha(:,ind);
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
function [likeval,alpha] = fitfun_class(hypx,y,x,alpha)
%hypx-超参数(mean,sn,sf,ell) y-响应 x-因子
for i=1:size(hypx,1)
[K,~]= covSEfun2(hypx(i,2:end),x,x);   %协方差矩阵
% alpha = zeros(size(y,1),1);
m=ones(size(y,1),1).*hypx(i,1);
alpha(:,i) = Nt_iter_ls(alpha(:,i), m,K,y);
f = K*alpha(:,i)+m;   yf = y.*f;                            % product latents and labels
[lp,dlp,d2lp] = logphi(yf); W = -d2lp; n = size(K,1); dlp = dlp.*y;   %logp(f|x,y)及导数
% symmetric B = I+sW*K*sW
sW = sqrt(W); 
L = chol(eye(n)+sW*sW'.*K,"lower");             % Cholesky factor of B
ldB2 = sum(log(diag(L)));                                    % log(det(B))/2
% compute marginal likelihood
likeval(i,1) = -alpha(:,i)'*(f-m)/2 + sum(lp) - ldB2;             %Laplace近似的似然函数，logq(y|X,θ) = -1/2f'inv(K)f+logp(y|f)-1/2log|B|
end
end
end