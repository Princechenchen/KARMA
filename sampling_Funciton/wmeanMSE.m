function [x_star, model] =  wmeanMSE(model)
ns = size(model.candidate,1); yd = size(model.init_value,2);
[y, s, model.MMRGP]=predict_resp(model.MMRGP,model.candidate);
for i=1:yd
    var(:,i) = s(i:yd:end,i);
end
weight = var./sum(var,2);
wmeanvar = sum(var.*weight,2);

[bestvalue,index] = max(wmeanvar);
x_star = model.candidate(index,:);
model.M_ind(size(model.now_value,1)-size(model.init_value,1)+1)=index;