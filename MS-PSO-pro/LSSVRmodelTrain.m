function [Y_train,error] = LSSVRmodelTrain(INPUT,OUTPUT,kernel,path)
%输出训练集预测结果及预测误差 [dmodel,Y_train,error] = mixed_response_modeling(INPUT,OUTPUT,ClassPos);
% INPUT=lhsdesign(40,2);
% OUTPUT=shubert_2d(INPUT);
% kernel=3;

[model,Y_train,error] = LSSVR_response_modeling(INPUT,OUTPUT,kernel);
% save('LSSVR_model.mat',"model");
separator = filesep;
% path="C:\Users\Passenger\Desktop\Model\LSSVR_response";
save(strcat(path,separator,'LSSVR_model.mat'),"model");
end