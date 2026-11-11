function [model,Y_train,error]=LSSVR_response_modeling(Input,Output,kernel)
% Input=lhsdesign(10,2);
% Output=shubert_d(Input);
% kernel=1;
trainX=Input;
trainY=Output;
x_max=max(trainX);x_min=min(trainX);
dimension_dx=size(trainX,2);
x_level=4*ones(1,dimension_dx);
% x_level(dimension_dx)=length(unique(trainX(:,dimension_dx)));
XX = gridsamp([x_min;x_max], x_level);
% XX(:,dimension_dx)=floor(XX(:,dimension_dx)) ;
XX=unique(XX,'rows');
type = 'function estimation';
% yx=shubert_2d(XX);
if kernel==1
    kernel_type='RBF_kernel';
    [gam,sig2] = tunelssvm({trainX,trainY,type,[],[],kernel_type},'simplex','leaveoneoutlssvm',{'mse'});
    %'crossvalidatelssvm',{10,'mse'}
    % [gam,sig2] = tunelssvm({points,values,type,[],[],'RBF_kernel'},'gridsearch','leaveoneoutlssvm',{'mse'});
    %%
    [alpha,b] = trainlssvm({trainX,trainY,type,gam,sig2,kernel_type});
    pre_y = simlssvm({trainX,trainY,type,gam,sig2,kernel_type,'preprocess'},{alpha,b},XX);
    Y_train = simlssvm({trainX,trainY,type,gam,sig2,kernel_type,'preprocess'},{alpha,b},trainX);
end
if kernel==2
    kernel_type='lin_kernel';
    sig2=0;
    [gam] = tunelssvm({trainX,trainY,type,[],[],kernel_type},'simplex','leaveoneoutlssvm',{'mse'});
    [alpha,b] = trainlssvm({trainX,trainY,type,gam,[],kernel_type});
    pre_y = simlssvm({trainX,trainY,type,gam,[],kernel_type,'preprocess'},{alpha,b},XX);
    Y_train = simlssvm({trainX,trainY,type,gam,[],kernel_type,'preprocess'},{alpha,b},trainX);
end
if kernel==3
    kernel_type='poly_kernel';
    [gam,sig2] = tunelssvm({trainX,trainY,type,[],[],kernel_type},'simplex','leaveoneoutlssvm',{'mse'});
    %'crossvalidatelssvm',{10,'mse'}
    % [gam,sig2] = tunelssvm({points,values,type,[],[],'RBF_kernel'},'gridsearch','leaveoneoutlssvm',{'mse'});
    %%
    [alpha,b] = trainlssvm({trainX,trainY,type,gam,sig2,kernel_type});
    pre_y = simlssvm({trainX,trainY,type,gam,sig2,kernel_type,'preprocess'},{alpha,b},XX);
    Y_train = simlssvm({trainX,trainY,type,gam,sig2,kernel_type,'preprocess'},{alpha,b},trainX);
end
model.kernel=kernel;
model.gam=gam;
model.sig2=sig2;
model.alpha=alpha;
model.b=b;
model.trainX=trainX;
model.trainY=trainY;
% XXY=[XX,pre_y];

Y0=[trainY,Y_train];
test_error = trainY - Y_train;
Y_mse = mse(test_error);
Y_rmse= sqrt(mean((test_error).^2));
Y_mape = sum ( abs( Y0(:,1)-Y0(:,2) )./Y0(:,1) )/size(Y0(:,1), 1);
error=[Y_mse,Y_rmse,Y_mape];
end