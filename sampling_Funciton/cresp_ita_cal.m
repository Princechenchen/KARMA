function delta_ita = cresp_ita_cal(YK, STDYK,candx,model)
yd = size(model.init_value,2);
%prior
[meanvalue0, stdvalue0,md_MMRGP]=predict_resp(model.MMRGP,model.testsample);
stdvalue0 = stdvalue0(find(model.ClassPos==2):yd:end,find(model.ClassPos==2));
uppervalue=md_MMRGP.fc+3*stdvalue0;
lowervalue=md_MMRGP.fc-3*stdvalue0;
ita0=size(find(lowervalue.*uppervalue<=0),1)/size(meanvalue0,1);
ita0=size(find(abs(md_MMRGP.fc)<=3*stdvalue0),1)/size(meanvalue0,1);

%nh: number of simulated samples to be the 100 percentiles of the kriging
%posterior
nh=100; ns = size(model.candidate,1);
% PDF=zeros(ns,nh);
% Hall=zeros(ns,nh);
% weights=zeros(ns,nh);

[x,w]=GaussHermite(nh);

Hall=(sqrt(2)*STDYK*x+YK)';
weights=(w/sqrt(pi))';

for j=1:size(Hall,2)
    y_star=Hall;
    Hmodel{j}=model;
    Hmodel_value=[Hmodel{j}.now_value;y_star(j)];
    Hmodel_x=[Hmodel{j}.now_x;candx];  
    Hmodel{j}.MMRGP = MMRGP(Hmodel_value,Hmodel_x,Hmodel{j}.ClassPos,Hmodel{j}.dim);
    [meanvalue, stdvalue,md_MMRGP] = predict(Hmodel{j}.MMRGP , model.testsample); %%
    uppervalue=md_MMRGP.fc+3*stdvalue;
    lowervalue=md_MMRGP.fc-3*stdvalue;
    itaM(j)=size(find(lowervalue.*uppervalue<=0),1)/size(meanvalue,1);
end
PDF=itaM.*weights;
delta_ita = ita0-sum(PDF,2);

