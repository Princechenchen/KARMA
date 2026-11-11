clear all
%% Design variables definition
% Generating sample data: 16 uniform distributed points is considered as
% the training data. This training data can be changed or chosen from any
% other algorithm like LHS
% n0      = 6;                             % number of samples along each direction (variable)
% input1  = linspace(0,100,n0);          % generating the uniform samples along the first direction
% input2  = linspace(0,100,n0);          % generating the uniform samples along the second direction
% [INPUT1,INPUT2] = meshgrid(input1,input2);
% INPUT11 = reshape(INPUT1,n0*n0,1);  % generating mesh grids of data (will be used for ploting)
% INPUT22 = reshape(INPUT2,n0*n0,1);  % generating mesh grids of data (will be used for ploting)
% INPUT   = [INPUT11,INPUT22];          % combine the two dimensions (variables)
% % OUTPUT=[bmfun1(INPUT),bmfun2(INPUT),bmfun3(INPUT)];
num = 30;  dm = 2; %训练集大小及维度
INPUT = lhsdesign(num,dm);
OUTPUT=[Branin_2dc(INPUT),Branin_2d(INPUT),Branin_2d2(INPUT)];
ClassPos = [1,0,0]; %%%处理分类响应  2维画图展示与多维通用分开

np = 10000;
t = lhsdesign(np,dm);
z_true=[Branin_2dc(t),Branin_2d(t),Branin_2d2(t)];
x=INPUT;n = length(t);
%% continuous and classification response hyp solve
for i=1:length(ClassPos)
    y=OUTPUT(:,i);
if ClassPos(i)==0
    meanfunc = @meanConst;                    % empty: don't use a mean function
    covfunc = @covSEiso;              % Squared Exponental covariance function   % k(x,z) = sf^2 * exp(-(x-z)'*inv(P)*(x-z)/2) 
    likfunc = @likGauss;              % Gaussian likelihood 
    hyp = struct('mean', 0, 'cov', [0.5 0.5], 'lik', -1);
    hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
    [mu s2, ~, ~, lp, post] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, t);
    [nlZ, dnlZ, post] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
    Hyp{i} = hyp2; lhv(i) = nlZ; ypred(:,i)=mu;
else
    meanfunc = @meanConst; hyp.mean = 0;
    covfunc = @covSEard; ell = 1.0; sf = 1.0; hyp.cov = log([ell ell sf]);
    likfunc = @likErf; 

    hyp = minimize(hyp, @gp, -40, @infLaplace, meanfunc, covfunc, likfunc, x, y);             %infEP后面再学
    [a b c d lp2] = gp(hyp, @infLaplace, meanfunc, covfunc, likfunc, x, y, t, ones(n, 1));
    [nlZ, dnlZ, post]= gp(hyp, @infLaplace, meanfunc, covfunc, likfunc, x, y);
    p = exp(lp2);          %预测概率  以0.5为界
    ymu = 2*p-1;          %分类响应预测均值   %以0为界，+1>0, -1<0
    ys2 = 4*p.*(1-p);     %预测方差
    ymu(ymu<0.5) = -1;
    ymu(ymu>=0.5) = 1;
    Hyp{i} = hyp; lhv(i) = nlZ;ypred(:,i)=ymu;
end
end
%% realpredict
% z_true=[bmfun1(t),bmfun2(t),bmfun3(t)];
% if dm==2 %2维画图
% np=100;
% [t1 t2] = meshgrid(linspace(0,1,np),linspace(0,1,np));
% t = [t1(:) t2(:)];                % these are the test inputs
% z_true=[beale_2d(t),Branin_2d(t),egg_2d(t)];  %%%处理分类响应
% X=t1;Y=t2;
% for b=1:size(OUTPUT,2)
% % b = 2;
% z_t = reshape(z_true(:,b),np,np);
% z_p = reshape(ypred(:,b),np,np);
% figure
% surf(X,Y,z_t);
% hold on
% plot3 (INPUT(:,1),INPUT(:,2),OUTPUT(:,b),'ob', 'MarkerFaceColor','b');
% xlabel('x1'); ylabel('x2'); zlabel('out');
% figure
% surf(X,Y,z_p');
% hold on
% plot3 (INPUT(:,1),INPUT(:,2),OUTPUT(:,b),'ob', 'MarkerFaceColor','b');
% xlabel('x1'); ylabel('x2'); zlabel('out');
% end
% end
%% predict error

% error = (((sum(sum(abs(z_true-ypred))))).^1)/(n*n);  % error calcualtion using MAE(mean absolute error)
% error = sum(error)/3;                                 % calculate the mean of error
error = error_func(z_true,ypred,ClassPos);
%% the objective functions... you can customize them and see the effect! 
function response = bmfun1(x)
response = (0.55+x(:,1).^2+x(:,2).^2).*( 1+sin(2*pi*(x(:,1)-0.25)).*sin(2*pi*(x(:,2)-0.25)) )-1;
response = (1+exp(-1.*response)).^-1;
response(response<0.5) = -1;
response(response>=0.5) = 1;
end
function response = bmfun2(x)
% response = 0+0.8*(1+x(:,1).^2+x(:,2).^2).*( 1+sin(2*pi*(x(:,1)-0.25)).*sin(2*pi*(x(:,2)-0.25)) );
% response = 10+0.5*(2+x(:,1).^2+x(:,2).^2).*( 0.5+sin(2*pi*(x(:,1)-0.5)).*sin(2*pi*(x(:,2)-0.25)) );
response = (0.55+x(:,1).^2+x(:,2).^2).*( 1+sin(2*pi*(x(:,1)-0.25)).*sin(2*pi*(x(:,2)-0.25)) )-1;
end
function response = bmfun3(x)
response = 0.8*(1+x(:,1).^2+x(:,2).^2).*( 1+sin(2*pi*(x(:,1)-0.25)).*sin(2*pi*(x(:,2)-0.25)) );
% response = (-1+2*x(:,1)+x(:,2).^2).*( 1+sin(2*pi*(x(:,1)-0.25)));
end