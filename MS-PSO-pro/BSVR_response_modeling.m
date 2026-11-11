function [model,Y_train,error]=BSVR_response_modeling(Input,Output,theta0,C0,lbt,ubt,lbc,ubc)
% clc; clear;
% Input=lhsdesign(40,6);
% Output=rastr_2d(Input);
% train=[Input,Output];
% theta0=10;C0=100;lbt=0.001;ubt=100;lbc=1;ubc=10000;
trainX=Input;
trainY=Output;
x_max=max(trainX);x_min=min(trainX);
dimension_dx=size(trainX,2);
x_level=5*ones(1,dimension_dx);
XX = gridsamp([x_min;x_max], x_level);
XX=unique(XX,'rows');
% yx=shubert_2d(XX);
d=size(trainX,2);
Hyperparameters=([C0  theta0.*ones(1,d)]);               %Initial values of Hyper-parameters
lb=([lbc     lbt.*ones(1,d)]);                      %lower bound of Hyper-parameters
ub=([ubc  ubt.*ones(1,d)]);    
model= SVRmodel(trainX,trainY,Hyperparameters,lb,ub);
[pre_y,~]=SVRPredict(XX,model);
XXY=[XX,pre_y];
[Y_train,~]=SVRPredict(trainX,model);
Y0=[trainY,Y_train];
test_error = trainY - Y_train;
Y_mse = mse(test_error);
Y_rmse= sqrt(mean((test_error).^2));
Y_mape = sum ( abs( Y0(:,1)-Y0(:,2) )./Y0(:,1) )/size(Y0(:,1), 1);
error=[Y_mse,Y_rmse,Y_mape];
end
