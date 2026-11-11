function [Y,Variance,model]=SVRPredict(X1,model)
%SVR prediction
[m n] =size(model.Input);  n1=size(X1,1);

beta=model.parameter;

MInput=model.Inputmoment(1,:);
SInput=model.Inputmoment(2,:);
Moutput=model.outputmoment(1,:);
Soutput=model.outputmoment(2,:);

X1=(X1-repmat(MInput,n1,1))./repmat(SInput,n1,1);

X=model.Input;   theta=model.theta;
%m是训练集样本点个数，n1是测试集样本点个数
mx=n1;
dx = zeros(mx*m,n);  kk = 1:m;
for  k = 1 : mx
    dx(kk,:) = repmat(X1(k,:),m,1) - X;
    kk = kk + m;
end
[m1 n] = size(dx);  % number of differences and dimension of data
if  length(theta) == 1
    theta = repmat(theta,1,n);
elseif  length(theta) ~= n
    error(sprintf('Length of theta must be 1 or %d',n))
end

td = dx.^2 .* repmat(-theta(:).',m1,1);
r = exp(sum(td, 2));
H= reshape(r, m, mx)';

Y=H*beta;
R1=model.UpperMatrix;
model.H=H;
%n1为测试集个数
for i=1:n1
    Variance(i)=1-diag(H(i,:)*(R1\(R1'\H(i,:)')));
end

Y=Y.*Soutput+Moutput;    %Prediction
Variance=Variance'.*Soutput.^2; %Prediction variance
end


