function [testX,predictY] = LSSVRmodelPredict(testX,modelFile)
% modelFile='LSSVR_model';
% sprintf(modelFile);
% testX=lhsdesign(40,2);
load(modelFile);
kernel=model.kernel;
gam=model.gam;
sig2=model.sig2;
alpha=model.alpha;
b=model.b;
trainX=model.trainX;
trainY=model.trainY;
type = 'function estimation';
% yx=shubert_2d(XX);
if kernel==1
    kernel_type='RBF_kernel';
    predictY = simlssvm({trainX,trainY,type,gam,sig2,kernel_type,'preprocess'},{alpha,b},testX);
end
if kernel==2
    kernel_type='lin_kernel';
    predictY = simlssvm({trainX,trainY,type,gam,[],kernel_type,'preprocess'},{alpha,b},testX);
end
if kernel==3
    kernel_type='poly_kernel';
    predictY = simlssvm({trainX,trainY,type,gam,sig2,kernel_type,'preprocess'},{alpha,b},testX);
end
% [testY,covmatrix,testY_sigma] = mixed_response_predict(testX,modelFile);
% predictY=[testX,testY];
% predictY=testY;
XXY=[testX,predictY];
end
