 clear all, close all
 
%   n1 = 80; n2 = 40;                   % number of data points from each class
%   S1 = eye(2); S2 = [1 0.95; 0.95 1];           % the two covariance matrices
%   m1 = [0.75; 0]; m2 = [-0.75; 0];                            % the two means
%   x1 = bsxfun(@plus, chol(S1)'*gpml_randn(0.2, 2, n1), m1);
%   x2 = bsxfun(@plus, chol(S2)'*gpml_randn(0.3, 2, n2), m2);
%   x = [x1 x2]'; y = [-ones(1,n1) ones(1,n2)]';

      n0      = 6;                             % number of samples along each direction (variable)
    input1  = linspace(0,1,n0);          % generating the uniform samples along the first direction
    input2  = linspace(0,1,n0);          % generating the uniform samples along the second direction
    [INPUT1,INPUT2] = meshgrid(input1,input2);
    INPUT11 = reshape(INPUT1,n0*n0,1);  % generating mesh grids of data (will be used for ploting)
    INPUT22 = reshape(INPUT2,n0*n0,1);  % generating mesh grids of data (will be used for ploting)
    x   = [INPUT11,INPUT22];          % combine the two dimensions (variables)  
    y = (0.55+x(:,1).^2+x(:,2).^2).*( 1+sin(2*pi*(x(:,1)-0.25)).*sin(2*pi*(x(:,2)-0.25)) )-1;
    y = (1+exp(-1.*y)).^-1;
    y(y<0.5) = -1;
    y(y>=0.5) = 1;
    x1 = x(find(y==1),:);x1=x1';
    x2 = x(find(y==-1),:);x2=x2';
    [t1 t2] = meshgrid(0:0.01:1,0:0.01:1);

%   plot(x1(1,:), x1(2,:), 'b+'); hold on;
%   plot(x2(1,:), x2(2,:), 'r+');

%   [t1 t2] = meshgrid(-4:0.1:4,-4:0.1:4);
  t = [t1(:) t2(:)]; n = length(t);                 % these are the test inputs
%   tmm = bsxfun(@minus, t, m1');
%   p1 = n1*exp(-sum(tmm*inv(S1).*tmm/2,2))/sqrt(det(S1));
%   tmm = bsxfun(@minus, t, m2');
%   p2 = n2*exp(-sum(tmm*inv(S2).*tmm/2,2))/sqrt(det(S2));
%   contour(t1, t2, reshape(p2./(p1+p2), size(t1)), [0.1:0.1:0.9]);

  meanfunc = @meanConst; hyp.mean = 0;
  covfunc = @covSEard; ell = 1.0; sf = 1.0; hyp.cov = log([ell ell sf]);
  likfunc = @likErf; 

  hyp = minimize(hyp, @gp, -40, @infLaplace, meanfunc, covfunc, likfunc, x, y);             %infEP后面再学
  rng(10);
  [a b c d lp2] = gp(hyp, @infLaplace, meanfunc, covfunc, likfunc, x, y, t, ones(n, 1));
  [nlZ, dnlZ, post]= gp(hyp, @infLaplace, meanfunc, covfunc, likfunc, x, y);
  figure()
  plot(x1(1,:), x1(2,:), 'b+'); hold on; plot(x2(1,:), x2(2,:), 'r+')
  contour(t1, t2, reshape(exp(lp2), size(t1)), [0.1:0.1:0.9]);


% hold on
% plot3 (x1(:,1),x1(:,2),ymu,'ob', 'MarkerFaceColor','b');
% xlabel('x1'); ylabel('x2'); zlabel('out');
  %%
hmean = hyp.mean;                  %均值
sn = 0;                     %随机误差
sf = exp(hyp.cov(end));     %协方差超参数
ell = exp(hyp.cov(1:end-1));%协方差超参数
hyp2 = [hmean,sn,sf,ell];
hyp2 = [0,0,1,1,1];
% alpha = zeros(size(y,1),1);
% [likeval,alpha] = fitfun_class(hyp2,y,x,alpha);

%优化超参数
% [hypsolve,likeval,alpha] = PSO_optimize_c(hyp2,y,x);
[hypsolve,likeval,alpha] = Grad_optimize_c(hyp2,y,x,-50);
hyp2 = hypsolve;
m=ones(size(y,1),1).*hyp2(1);
[K,~]= covSEfun2(hyp2(2:end),x,x);   %协方差矩阵
alpha = Nt_iter_ls(alpha, m,K,y);
f = K*alpha+m;   yf = y.*f;                            % product latents and labels
[lp,dlp,d2lp] = logphi(yf); W = -d2lp; n = size(K,1); dlp = dlp.*y;   %logp(f|x,y)及导数
% symmetric B = I+sW*K*sW
sW = sqrt(W); 
L = chol(eye(n)+sW*sW'.*K,"lower");             % Cholesky factor of B
ldB2 = sum(log(diag(L)));                                    % log(det(B))/2
% compute marginal likelihood
likehoodval = -alpha'*(f-m)/2 + sum(lp) - ldB2;             %Laplace近似的似然函数，logq(y|X,θ) = -1/2f'inv(K)f+logp(y|f)-1/2log|B|

% Psi_new = Newton_target(alpha,m,K,y);

%预测潜变量mean   
xtest = t;
muytest = ones(size(xtest,1),1).*hyp2(1); 
[Kfs,~] = covSEfun2(hyp2(2:end),x,xtest);
fmu = Kfs'*alpha+muytest;   %repmat(mu,1,size(alpha,2)) + Ks'*full(alpha(nz,:)); 
% fmu = m + Kfs'*alpha;        % conditional mean fs|f
%预测潜变量variance
v = L\(sW.*Kfs);
[kss,~]= covSEfun2(hyp2(2:end),xtest,xtest); 
fs2 = diag(kss) - sum(v.*v,1)';  % fs2 = diag(kss) - diag(v'*v);
% fs2 =diag(kss)-sum(Kfs.*(L'\(L\(Kfs.*sW)).*sW),1)';
%预测分类响应
z = fmu./sqrt(1+fs2);
lZ = logphi(z);       %带入压缩函数
p = exp(lZ);          %预测概率  以0.5为界
ymu = 2*p-1;          %分类响应预测均值   %以0为界，+1>0, -1<0
ys2 = 4*p.*(1-p);     %预测方差

ymu(ymu<0.5) = -1;
ymu(ymu>=0.5) = 1;


figure()
plot(x1(1,:), x1(2,:), 'b+'); hold on; plot(x2(1,:), x2(2,:), 'r+')
contour(t1, t2, reshape(exp(lZ), size(t1)), [0.1:0.1:0.9]);

n = 101;
x = linspace(0,1,n);  % generating testing points uniformly along the firt direction
y = linspace(0,1,n);  % generating testing points uniformly along the second direction
[X,Y] = meshgrid(x,y);  % generate the 2d samples by combining the generated unidirectional points
z_t = reshape(ymu,n,n);
figure
surf(X,Y,z_t');
% hold on
% plot3 (x1(:,1),x2(:,2),ymu,'ob', 'MarkerFaceColor','b');
% xlabel('x1'); ylabel('x2'); zlabel('out');
p = exp(lp2);          %预测概率  以0.5为界
ymu = 2*p-1;          %分类响应预测均值   %以0为界，+1>0, -1<0
ys2 = 4*p.*(1-p);     %预测方差

ymu(ymu<0.5) = -1;
ymu(ymu>=0.5) = 1;
n = 101;
x = linspace(0,1,n);  % generating testing points uniformly along the firt direction
y = linspace(0,1,n);  % generating testing points uniformly along the second direction
[X,Y] = meshgrid(x,y);  % generate the 2d samples by combining the generated unidirectional points
z_t = reshape(ymu,n,n);
figure
surf(X,Y,z_t');
