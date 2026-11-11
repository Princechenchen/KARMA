function [mu_z,var_z,p,fmu,var_f] = mmrgp_predict_all_latest(query_x,model_id)
% Full mixed-response prediction contract for formal folded experiments.
assert(~isempty(regexp(model_id,'^[A-Za-z0-9_]+$','once')),'Invalid model_id');
name=['crimes_mmrgp_',model_id];
assert(evalin('base',sprintf('exist(''%s'',''var'')',name))==1,'Model is not fitted');
model=evalin('base',name);
[y,s,pm]=predict_resp(model,query_x);
n=size(query_x,1); m=model.m;
cont=find(model.classpos==1,1,'first'); cls=find(model.classpos==2,1,'first');
assert(cont==1 && cls==2,'Crimes response layout must be continuous then binary');
ci=cont:m:(n*m); ki=cls:m:(n*m);
cont_idx=sub2ind(size(s),ci,repmat(cont,size(ci)));
cls_idx=sub2ind(size(s),ki,repmat(cls,size(ki)));
scale=(model.settings.xmax(cont)-model.settings.xmin(cont))/2;
mu_z=y(:,cont); var_z=reshape(s(cont_idx)*scale^2,[],1);
p=(pm.fc(:)+1)/2; p=min(max(p,0),1);
fmu=pm.fmu(:); var_f=reshape(s(cls_idx),[],1);
end
