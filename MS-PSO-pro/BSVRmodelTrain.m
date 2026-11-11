function [Y_train,error] = BSVRmodelTrain(INPUT,OUTPUT,theta0,C0,lbt,ubt,lbc,ubc,path)
%输出训练集预测结果及预测误差 [dmodel,Y_train,error] = mixed_response_modeling(INPUT,OUTPUT,ClassPos);
% INPUT=lhsdesign(40,6);
% OUTPUT=rastr_2d(INPUT);
% theta0=10;C0=100;lbt=0.001;ubt=100;lbc=1;ubc=10000;
[model,Y_train,error] = BSVR_response_modeling(INPUT,OUTPUT,theta0,C0,lbt,ubt,lbc,ubc);
% separator = filesep;
% mkdir(path)
% save('BSVR_model.mat',"model");
separator = filesep;
% path="C:\Users\Passenger\Desktop\Model\BSVR_response";
save(strcat(path,separator,'BSVR_model.mat'),"model");
end