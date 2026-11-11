function model= SVRTrain(X,Y,Hyperparameters)
%Training Bayesian least square SVR model
[m n] = size(X);  % number of design sites and their dimension
MInput = mean(X);   SInput= std(X); %Normalization of data
X= (X-repmat( MInput,m,1))./repmat(SInput,m,1);
Moutput=mean(Y); Soutput=std(Y);
Y=(Y-repmat(Moutput,m,1))./repmat(Soutput,m,1);

Hyperparameters=exp(Hyperparameters); %Hyper-parameters
C=Hyperparameters(1);                 %Trade-off parameter
theta=Hyperparameters(2:end);         %Covariance function hyper-parameters

H=SVRcorrgauss(theta,X);              %Upper triangular Covariance matrix

H1=full(H);  H=H1+H1'-diag(ones(1,m)); %Full Covariance matrix
%Kernel=K+IC
Kernel=H+diag(1./C.*ones(m,1));        %Regularized Covariance matrix正则化相关矩阵

[R1 rd]=chol(Kernel);  %Cholskey decomposition分解
beta=R1\(R1'\Y);                        %SVR coefficients

model.Input=X;
model.Output=Y;
model.parameter=beta;
model.Kernel='Gaussian';
model.C=C;
model.UpperMatrix=R1;
model.Kernelmatrix=Kernel;
model.Inputmoment=[ MInput; SInput];
model.outputmoment=[Moutput; Soutput];
model.theta=theta;
end






