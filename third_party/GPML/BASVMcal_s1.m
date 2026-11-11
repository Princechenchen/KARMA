function errors=BASVMcal_s1(Input,Output,X0)
% Input=standard(Input);
% Output=standard(Output);
d=size(Input,2);
Hyperparameters=([100  10.*ones(1,d)]);               %Initial values of Hyper-parameters
lb=([1     0.001.*ones(1,d)]);                      %lower bound of Hyper-parameters
ub=([10^6  1000.*ones(1,d)]);     
% [Input,settings1]=mapminmax(Input',0,1);
% Input=Input';%upper bound of Hyper-parameters
% [Output,settings2]=mapminmax(Output',0,1);
% Output=Output';
model= SVRmodel(Input,Output,Hyperparameters,lb,ub);
%% Sasena_2d
% range = [0 : 0.7 : 5];
% X0 = makeEvalGrid({range, range});
% F0=Sasena_2d(X0(:,1),X0(:,2));
%% zhou98_d
% range = [-2 : 0.05 : 2];
% range1=[-1:0.03:1];
% X0 = makeEvalGrid({range, range1}); % generate a dense 2D grid
% F0=zhou98_d(X0(:,1),X0(:,2));
%% ishi_3d
% range1=[-1:0.027:1];
% X0 = makeEvalGrid({range1, range1}); % generate a dense 2D grid
% F0=ishi_3d(X0);
%% Eggholder_2d
% range1=[-200:5:200];
% X0 = makeEvalGrid({range1, range1}); % generate a dense 2D grid
% F0=Eggholder_2d(X0);
%% ishi_3d
% range1=[-2.5:0.02:2.5];
% X0 = makeEvalGrid({range1, range1}); % generate a dense 2D grid
% F0=ishi_3d(X0);

% range = [-2 : 0.05 : 2];
% range1=[-1:0.03:1];
% range2 = [-3 : .05 : 3];
% X0 = makeEvalGrid({range, range1}); % generate a dense 2D grid
%% Shubert
% range = [-2 : 0.04 : 1];
% X0 = makeEvalGrid({range, range}); % generate a dense 2D grid
% F0=shubert_2d(X0(:,1),X0(:,2));
%% Shubert
% range = [-2 : 0.06 : 2];
% X0 = makeEvalGrid({range, range}); % generate a dense 
% F0=shubert_2d(X0(:,1),X0(:,2));
%% peaks
% range = [-5 : 0.15 : 5];
% X0 = makeEvalGrid({range, range}); % generate a dense 
% F0=peaks(X0(:,1),X0(:,2));
%% Hart3
% range = [0 : 0.06 : 1];
% X0 = makeEvalGrid({range, range,range}); % generate a dense 2D grid
% F0=hart3(X0(:,1),X0(:,2),X0(:,3));
%% rastar
% range = [-1 : 0.04 : 2];
% X0 = makeEvalGrid({range, range}); % generate a dense 
% F0=rastr_2d(X0);
%%
% X1=mapminmax('apply',X0',settings1);
% X1=X1';
[Y,~]=SVRPredict(X0,model);
% F0=Colville_4d(X0);
% F = F0(find(F0~=0));
% YX = YX(find(F0~=0));
% X=Colville_4d(X0(:,1),X0(:,2),X0(:,3),X0(:,4));
% F0=SHCamel_2d(X0(:,1),X0(:,2));
% F0=peaks(X0(:,1),X0(:,2));
F0=levy_6d(X0);
% F0=mapminmax('apply',F0',settings2);
% F0=F0';
% F0(F0<10^(-4))=0;
% F0=shubert_d(X0);
% F0=himmelblau_2d(X0);
X = F0((F0~=0));
Y = Y((F0~=0));
function error_val = standard(X)
% root mean squared error
m = numel(X);
mean_response = (1/m)*(sum(X));
error_val = sqrt((1/(m))*(sum((X-mean_response).^2)));
end

function error_val = RMAE(X,Y)
% relative maximum absolute error
error_val = max(abs(X-Y)); %/standard(X);

error_val = error_val/(max(X)-min(X));
end


function error_val = MeanAE(X,Y)
% mean absolute error
error_val = mean(abs(X-Y));

error_val = error_val/(max(X)-min(X));
end


function error_val = RMSE(X,Y)
% root mean squared error
error_val = sqrt(mean(((X-Y).^2)));

error_val = error_val/(max(X)-min(X));

end


function error_val = R_sq(X,Y)
% R_sq score
mean_response = mean(X);
MSE_val = (sum((X-Y).^2));

divisor = sum((X-mean_response).^2);

error_val = 1-(MSE_val/divisor);
end


function error_val=MAPE(X,Y)
    error_val = mean(abs((Y - X)./X));
end
% X=standard(X);
% Y=standard(Y);
errors=[RMAE(X,Y),MeanAE(X,Y),RMSE(X,Y),R_sq(X,Y),MAPE((X),(Y)),MAPE(standard(X),standard(Y))];
end

