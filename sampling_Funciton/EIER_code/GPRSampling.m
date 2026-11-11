function GPRSamp = GPRSampling(GPRmodel,InvK,Xtrain,Ytrain,XSampPool,NGPR)
   Ntrain = length(Ytrain);
   classpos = GPRmodel.classpos;
   [Meany, s, model]=predict_resp(GPRmodel,XSampPool);
   Meany(:,classpos==2) = model.fc;
   Meany = reshape(Meany',GPRmodel.m*size(XSampPool,1),1);
%    K = GPRmodel.cov_model(model.hyper.teta(classpos==2,:), model.X, model.X);
%    K = GPRmodel.zmn; K = K(dc:GPRmodel.m:end,dc:GPRmodel.m:end);
%    InvK=(linsolve(K,diag(ones(1,Ntrain))))';
   r_XSamp_Xtrain=define_covmatrix(GPRmodel,XSampPool,GPRmodel.X);  %cov:candX_X  训练样本和测试样本间的协方差向量
   FXtrain_big = ones(GPRmodel.m*GPRmodel.n,NGPR);
   Ftrain_big = ones(GPRmodel.m*size(XSampPool,1),NGPR);
   for i=1:GPRmodel.m
       % define the covariance function
       corr.name = 'gauss';
       corr.c0 = 1./GPRmodel.hyper.teta(i,:);%Sigma1.^2
       corr.sigma = 1;
    
       % Sample from the posterior GP using 
       [FXtrain,~] = randomfield(corr,Xtrain,'nsamples',NGPR,'filter', 0.95);
       [Ftrain,~] = randomfield(corr,XSampPool,'nsamples',NGPR,'filter', 0.95);
       FXtrain_big(i:GPRmodel.m:end,:)=FXtrain;
       Ftrain_big(i:GPRmodel.m:end,:)=Ftrain;
   end
   

   GPRSamp = Meany - r_XSamp_Xtrain*InvK*FXtrain_big+Ftrain_big;
end