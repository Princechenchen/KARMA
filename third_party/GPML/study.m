clear


% n = 40;
% x = gpml_randn(0.1, n, 1);                 % 20 training inputs
% y = sin(5*x) + 0.1*gpml_randn(0.9, n, 1);  % 20 noisy training targets
% % y = x(:,3).^2.*sin(2*x(:,1))+cos(x(:,2)) + 0.2*gpml_randn(0.9, n, 1);
% %测试集
% xtest = linspace(-3, 3, 61)';                  % 61 test inputs
% ytest = sin(5*xtest) + 0.1*gpml_randn(0.9, 61, 1);

n0      = 6;                             % number of samples along each direction (variable)
input1  = linspace(0,1,n0);          % generating the uniform samples along the first direction
input2  = linspace(0,1,n0);          % generating the uniform samples along the second direction
[INPUT1,INPUT2] = meshgrid(input1,input2);
INPUT11 = reshape(INPUT1,n0*n0,1);  % generating mesh grids of data (will be used for ploting)
INPUT22 = reshape(INPUT2,n0*n0,1);  % generating mesh grids of data (will be used for ploting)
INPUT   = [INPUT11,INPUT22];          % combine the two dimensions (variables)
x=INPUT;
y = (0.55+x(:,1).^2+x(:,2).^2).*( 1+sin(2*pi*(x(:,1)-0.25)).*sin(2*pi*(x(:,2)-0.25)) )-1;

np = 100;
[t1 t2] = meshgrid(linspace(0,1,np),linspace(0,1,np));
t = [t1(:) t2(:)];                % these are the test inputs
ytest = (0.55+t(:,1).^2+t(:,2).^2).*( 1+sin(2*pi*(t(:,1)-0.25)).*sin(2*pi*(t(:,2)-0.25)) )-1;
xtest = t;
% xtest = apxGrid('expand',{linspace(-3,3,60)',linspace(-3,3,60)',linspace(-3,3,60)'});
% xtest = lhsdesign(1000,3).*6-3;
% xtest = randn(60,2);
% ytest = xtest(:,3).^2.*sin(2*xtest(:,1))+cos(xtest(:,2)) + 0.2*gpml_randn(0.9, size(xtest,1), 1);
% xtest = [linspace(-1.9, 1.9, 101)]';
% Ktest = feval(covfunc{:}, hyp.cov, xtest);
% mutest = feval(meanfunc{:}, hyp.mean, xtest);
% ytest = chol(Ktest)'*gpml_randn(0.15, size(xtest,1), 1) + mutest + exp(hyp.lik)*gpml_randn(0.2, size(xtest,1), 1);
%%自带
meanfunc = @meanConst;                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function   % k(x,z) = sf^2 * exp(-(x-z)'*inv(P)*(x-z)/2) 
likfunc = @likGauss;              % Gaussian likelihood 
hyp = struct('mean', 0, 'cov', [0 0 ], 'lik', 1);
hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
[mu s2, ~, ~, lp, post] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xtest);
[nlZ, dnlZ, post] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
% figure(1)
% f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
% fill([xtest; flipdim(xtest,1)], f, [7 7 7]/8)
% hold on; plot(xtest, mu); plot(x, y, '+');plot(xtest, ytest);

X=t1;Y=t2;
z_t = reshape(ytest,np,np);
z_p = reshape(mu,np,np);
figure
surf(X,Y,z_t');
hold on
plot3 (INPUT(:,1),INPUT(:,2),y,'ob', 'MarkerFaceColor','b');
xlabel('x1'); ylabel('x2'); zlabel('out');
figure
surf(X,Y,z_p');
hold on
plot3 (INPUT(:,1),INPUT(:,2),y,'ob', 'MarkerFaceColor','b');
xlabel('x1'); ylabel('x2'); zlabel('out');

%超参数
sn= exp(-2.3824);     %噪声误差
ell = exp(-0.6352);   %协方差超参数-分母
sf = exp(-0.1045);    %协方差超参数-系数
hyp.cov = [exp(-0.6352) exp(-0.1045)]; hyp.mean = [0; 0;0]; hyp.lik = sn;
% hyp.cov = [exp(hyp2.cov(1)) exp(hyp2.cov(2))]; hyp.mean = [0; 0;0]; hyp.lik = exp(hyp2.lik);
hyp = [sn,sf,ell];
hyp = zeros(1,size(x,2)+2);
[hypsolve,likeval] = PSO_optimize2(hyp,y,x);
% [K,dist_matrix]= covSEfun(hyp,x,x);   %协方差矩阵
% hyp.cov = hypsolve(1:2);
% hyp.lik = hypsolve(3);
hyp = hypsolve;
[K,dist_matrix]= covSEfun2(hyp,x,x);   %协方差矩阵
% L = chol(eye(n)+1/sn.^2.*K)
sn = hyp(1);
L = chol(K+sn.^2.*eye(n),'lower');  %cholesky分解，K求逆
alpha = L'\(L\y);
%预测mean
muytest = zeros(size(xtest,1),1); 
[Kfs,~] = covSEfun2(hyp,x,xtest);
fmu = Kfs'*alpha;   %repmat(mu,1,size(alpha,2)) + Ks'*full(alpha(nz,:)); 
%预测variance
v = L\Kfs;
[kss,~]= covSEfun2(hyp,xtest,xtest);  
fs2 = diag(kss) - sum(v.*v,1)';  % fs2 = diag(kss) - diag(v'*v);
%计算边缘似然值log marginal likelihood
muy = zeros(size(y,1),1); 
%log(p(y|x)) = -1/2*y'*alpha-sum(diag(log(L)))-n/2log2pi
mar_likehoodval = -1/2*(y-muy)'*alpha-sum(diag(log(L)))-numel(y)/2*log(2*pi); 
% hypd = D_likelihood(x,y,alpha,L,K,hyp,dist_matrix);
%输出
ymu = fmu;
ys2 = fs2+sn^2;   %why

figure(2)
set(gca, 'FontSize', 24)
f = [ymu+2*sqrt(ys2); flipdim(ymu-2*sqrt(ys2),1)];
fill([xtest; flipdim(xtest,1)], f, [7 7 7]/8);
hold on; plot(xtest, ymu, 'LineWidth', 2); plot(x, y, '+', 'MarkerSize', 12)
plot(xtest, ytest);
grid on
xlabel('input, x')
ylabel('output, y')

%计算误差
errorval1 = errorfunc(ytest,mu,max(y)-min(y),size(xtest,2));
errorval2 = errorfunc(ytest,ymu,max(y)-min(y),size(xtest,2));






