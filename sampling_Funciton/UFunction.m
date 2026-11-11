function [x_star, model] =  UFunction(model)
ns = size(model.candidate,1); yd = size(model.init_value,2);
[y, s, model.MMRGP]=predict_resp(model.MMRGP,model.candidate);
for i=1:yd
    var(:,i) = s(i:yd:end,i);
end
yc = model.MMRGP.fc;
varc = var(:,model.ClassPos==2);
U = abs(yc)./varc;

[bestvalue,index] = min(U);
x_star = model.candidate(index,:);
model.M_ind(size(model.now_value,1)-size(model.init_value,1)+1)=index;
% for i=1:ns
%     [y, s, model.MMRGP]=predict_resp(model.MMRGP,model.candidate(i,:));
%     
% end