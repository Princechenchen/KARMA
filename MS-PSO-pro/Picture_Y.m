
%代理模型生成
Data2=xlsread('E:\研究\硕士毕设\代码\matlab版本2.0\修改\间歇压制+角反质心+距离拖引.xlsx','三组合筛选前12因子');
test_error=1;
factor=11;
%error_list=string_name;
x=Data2(:,1:factor);
y=Data2(:, factor+test_error);
[x,settings1]=mapminmax(x',0,1);
[y,settings2]=mapminmax(y',0,1);
input=x';
output=y';
rng('default') % For reproducibility
d=size(input,2);
Hyperparameters=([100  1.*ones(1,d)]);               %Initial values of Hyper-parameters
lb=([1     0.0001.*ones(1,d)]);                      %lower bound of Hyper-parameters
ub=([10^7  1000.*ones(1,d)]);                        %upper bound of Hyper-parameters
model= SVRmodel(input,output,Hyperparameters,lb,ub);
%Y=SVRPredict([0,0,0,0],model)


mean_vals = mean(input, 1); % 1 x factor

%可视化三个维度
x1_min = 0; x1_max = 1;
x2_min = 0; x2_max = 1;
x3_min = 0; x3_max = 1;%

% 创建网格点（仅前三个维度变化）
[x1, x2, x3] = ndgrid(linspace(x1_min, x1_max, 20), ...
                      linspace(x2_min, x2_max, 20), ...
                      linspace(x3_min, x3_max, 20));

% 创建输入点数组：前三维为网格点，其他维度固定为均值
input_points = zeros(size(x1, 1) * size(x1, 2) * size(x1, 3), factor);
input_points(:, 1) = x1(:);
input_points(:, 2) = x2(:);
input_points(:, 3) = x3(:);
for i = 4:factor
    input_points(:, i) = mean_vals(i);
end
output_values = SVRPredict(input_points,model);%使用代理模型预测y

% 可视化：使用散点图表示
figure;
scatter3(input_points(:,1), input_points(:,2), input_points(:,3), 80, output_values, 'filled');
colorbar; % 显示颜色条
% xlabel('干信比');
% ylabel('间断周期');
% zlabel('占空比');
% title('角跟踪精度');
xlim([x1_min-0.05, x1_max+0.05]);
ylim([x2_min-0.05, x2_max+0.05]);
zlim([x3_min-0.05, x3_max+0.05]);

xlabel('压制干信比');
ylabel('压制占空比');
zlabel('压制结束时刻');
title('跟踪假目标时长占比');
