% gp_example  gp_single 完整使用示例(纯建模+预测,无画图)
% 运行(在本目录下): matlab -batch gp_example
% 内容:1) gp_train 建模  2) gp_predict 预测  3) gp_loo 评估  4) 重复测量数据示例
clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(here);
root = fileparts(here);                                    % 项目根
if exist(fullfile(root,'testFunc','Branin_2d3.m'),'file')
    addpath(fullfile(root,'testFunc'));
    ftrue = @(X) log(Branin_2d3(X));                       % 项目自带测试函数
else
    ftrue = @(X) sin(4*pi*X(:,1)).*cos(3*pi*X(:,2)) + 0.5*(X(:,1)-0.5).^2;
end

%% ===== 1) 建模:gp_train(X, y) =====
rng(1);
X = rand(40,2);                          % 训练输入 n×d(每行一个样本)
y = ftrue(X);                            % 训练输出 n×1(单响应)

model = gp_train(X, y, 'cov','gpCovSEard', 'nrestart',5, 'seed',1, 'verbose',true);

fprintf('\n--- 训练结果 ---\n');
fprintf('nlml(raw) = %.3f(负对数边际似然,越小越好)\n', model.nlml_raw);
fprintf('长度尺度 l  = [%s]\n', num2str(exp(model.hyp_raw.cov(1:end-1)).','%.3f '));
fprintf('信号强度 sf = %.3f  噪声 sn = %.4f\n', exp(model.hyp_raw.cov(end)), exp(model.hyp_raw.lik));

%% ===== 2) 预测:gp_predict(model, Xs) =====
rng(2);
Xt = rand(300,2);                        % 测试输入 m×d(任意点,可为训练点本身)
yt = ftrue(Xt);                          % 真值,仅用于评估

[mu, s2, ci] = gp_predict(model, Xt);    % 潜在函数口径:均值/方差/95%CI
[muN, s2N]   = gp_predict(model, Xt, true);   % 含噪口径(可选)

fprintf('\n--- 预测质量 ---\n');
fprintf('test RMSE = %.4f\n', sqrt(mean((yt - mu).^2)));
fprintf('95%% CI 覆盖率 = %.1f %%\n', 100*mean(yt >= ci(:,1) & yt <= ci(:,2)));
fprintf('后验标准差范围 = [%.3f, %.3f]\n', min(sqrt(s2)), max(sqrt(s2)));
fprintf('含噪方差-潜在方差 = sn^2 = %.5f(自检应≈sn^2)\n', mean(s2N - s2));

%% ===== 3) 评估:gp_loo(model)(可选,一行) =====
[loo_rmse, loo_mlpd] = gp_loo(model);
fprintf('\nLOO RMSE = %.4f | 平均对数预测密度 = %.3f\n', loo_rmse, loo_mlpd);

%% ===== 4) 重复测量:同一输入多次测量,直接堆行,无需去重/预平均 =====
rng(3);
Xu = rand(15,2);                         % 15 个不同输入
Xr = repelem(Xu, 3, 1);                  % 每点重复 3 次 → 45×2
yr = ftrue(Xr) + 0.2*randn(45,1);        % 测量噪声 sd = 0.2

mrep = gp_train(Xr, yr);
fprintf('\n--- 重复测量 ---\n');
fprintf('真噪声 0.2000 | 估出 sn = %.4f\n', exp(mrep.hyp_raw.lik));
fprintf('EXAMPLE DONE OK\n');
