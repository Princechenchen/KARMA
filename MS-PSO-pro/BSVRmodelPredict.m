function [XXY,Variance] = BSVRmodelPredict(testX,modelFile)
% modelFile='BSVR_model';
% sprintf(modelFile);
% model=load(modelFile); 
% testX=lhsdesign(40,6);
model=load(modelFile); 
[predictY,Variance]=SVRPredict(testX,model.model);
% [testY,covmatrix,testY_sigma] = mixed_response_predict(testX,modelFile);
% predictY=[testX,testY];
% predictY=testY;
XXY=[testX,predictY];
end
