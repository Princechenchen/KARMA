function [x_star, model] =  SCF(model)
ns = size(model.candidate,1); yd = size(model.init_value,2);
[y, s, model.MMRGP]=predict_resp(model.MMRGP,model.candidate);
for i=1:yd
    var(:,i) = s(i:yd:end,i);
end

distvar = dist(model.candidate,model.init_x');
distvar = min(distvar,[],2);

meanvar = mean(var); maxvar=max(var);
weight = 0.5*(meanvar./sum(meanvar)+maxvar./sum(maxvar));
C = distvar.*sum(var.*weight,2);

[bestvalue,index] = max(C);
x_star = model.candidate(index,:);
model.M_ind(size(model.now_value,1)-size(model.init_value,1)+1)=index;