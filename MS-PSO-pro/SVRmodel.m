function model= SVRmodel(X,Y,Hyperparameters,lb,ub)   
%Optimizate the hyperparameters
 Hyperparameters=log(Hyperparameters);  lb=log(lb);   ub=log(ub);
%----------------------Fmincon
%创建优化选项，其中fmincom表示约束求解器，
 options=optimoptions('fmincon','Display','off'); 
 %FMINCON 查找多个变量函数的约束最小值。
 [Ohp,Of]=fmincon(@SVRLikelihood,Hyperparameters,[],[],[],[],lb,ub,[],options);
 model=SVRTrain(X,Y,Ohp);
%-------------------------GA
%  [Ohp,Of]=ga(@SVRLikelihood,size(Hyperparameters,2),[],[],[],[],lb,ub);
%  model=SVRTrain(X,Y,Ohp);
%-------------------------
 function Loglikelihood=SVRLikelihood(Hyperparameters) 
  %Likelihood function
  
  model= SVRTrain(X,Y,Hyperparameters);
  HP=exp(Hyperparameters); 
  n=size(X,1);
  Kernel=model.Kernelmatrix;
  parameter=model.parameter;
  Fmp=Kernel*parameter;
  Deta=model.Output-Fmp;
  C=model.C;
  
  theta=HP(2:end);
  Loss=0.5*Deta.^2;

  Zd=0.5*(n*log(C)+log(det(Kernel)));
  
  Loglikelihood=(0.5.*parameter'*Kernel*parameter+C*sum(Loss)+n*log((2.*pi./C).^0.5)+Zd); %Likelihood function
  
end    
end 
     
     
     
     
     
     
     