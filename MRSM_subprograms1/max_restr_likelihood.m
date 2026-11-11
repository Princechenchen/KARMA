function model= max_restr_likelihood(model)

% This function is used when we want to tune all hyperparameters simultaneously.
% To find the global optimum point we use Globalsearch function in Matlab.

MLE = @(x) rest_log_likelihood(x,model);      % define the likelihood function

% Set the consitraints used in optimization
A = [];
b = [];
Aeq = [];
beq = [];

%  Inital sets for parametrs of  z0
n1=model.m*(model.m-1)/2;
lb1= zeros(1,n1);
ub1= model.s*ones(1,n1);
x01 = 0.1*ones(1,n1);
% z0 = corrcoef(model.Y);
% k=1;
% for i=1: model.m
%     for j=i+1:model.m
%     x01(k)=abs(z0(i,j));
%     k=k+1;
%     end
% end 
% lb1=x01-0.3;
% ub1=x01+0.3; ub1(ub1>1)=1;
% x01 = 0.5*ones(1,n1);
% Initial sets for parameters of  teta
n2=model.m*model.d;
lb2= 0.001*ones(1,n2);
ub2= 15*ones(1,n2);   %%有修改(似乎有助于提升模型稳定性)
x02 =0.1*ones(1,n2);

% Combining all parameters:
n=n1+n2;
lb=[lb1,lb2];
ub=[ub1,ub2] ;
x0 =[x01,x02];

%% Optimization
% There are two options for optimization: 1- using direct fmincon function
% based on a single inital point x0. This is a faster method but can only
% find the local minima. 2- Using global search which can find the global
% minima. This method is more accurate but takes more time. 

if strcmp(model.optim,'fmincon')     % Option 1:
    options = optimset('Display', 'off') ;
    nonlcon=[];
    x= fmincon(MLE,x0,A,b,Aeq,beq,lb,ub, nonlcon, options);
else   % Option 2:
    opts = optimoptions(@fmincon,'Algorithm','interior-point');
    problem = createOptimProblem('fmincon','objective',MLE,'x0',x0,'lb',lb,'ub',ub,'options',opts);
    gs = GlobalSearch('NumTrialPoints',300);
    [x,f] = run(gs,problem);
end
        
%% Update hyperparametrs
% Update the matrix A
k=1; z0=[];
for i=1: model.m
    for j=i+1:model.m
        z0(i,j)=x(k);
        k=k+1;
    end
end
if model.m==1;
    z0=[];
end

z0=[z0;zeros(1,model.m)];
z0=z0+z0'+diag(ones(model.m,1));
[V, D]=eig(z0);
D(D<0)=0;
model.hyper.A=V*(D^0.5)*V';

% Update the hyperparameters teta
teta=x(n1+1:end);
teta=reshape(teta,model.m,model.d);
model.hyper.teta=teta;

end