function [x_star, model] = Voronoi_MixedMetric(model)
% Voronoi_MixedMetric  C10 Voronoi-Pareto-TOPSIS mixed-metric sampler.
% Complete MMRGP-adapted implementation of Zhao et al. (JME, 2024).
% Sequence: Monte-Carlo Voronoi cells, Delaunay neighbours, CS/LOOCV/NonL,
% Pareto filtering, entropy-TOPSIS, and maximum variance within selected cell.

Xcand = model.candidate;
X = model.now_x;
if isfield(model,'now_value') && ~isempty(model.now_value), Y=model.now_value; else, Y=model.init_value; end
n=size(X,1); d=size(X,2); m=size(Y,2);
if n<2 || size(Y,1)~=n, error('Voronoi_MixedMetric:InvalidTrainingData','now_x and response rows must align and contain at least two points.'); end

mc=Xcand;
if isfield(model,'c10_mc_samples') && model.c10_mc_samples>size(mc,1)
    lo=min([X;Xcand],[],1); hi=max([X;Xcand],[],1); span=hi-lo; span(span<=0)=1;
    old=rng; if isfield(model,'c10_seed') && ~isempty(model.c10_seed), rng(model.c10_seed,'twister'); end
    mc=lo+rand(model.c10_mc_samples,d).*span; rng(old);
end
[cell_id,dist_to_sample]=nearest_cell(mc,X);
CS=accumarray(cell_id,1,[n,1],@sum,0)/max(size(mc,1),1);
neighbours=delaunay_neighbours(X);

loo_exact=true; if isfield(model,'c10_loo_exact'), loo_exact=logical(model.c10_loo_exact); end
loo_cap=inf; if isfield(model,'c10_loo_max_points') && ~isempty(model.c10_loo_max_points), loo_cap=model.c10_loo_max_points; end
eLOOCV=loo_errors(model,X,Y,loo_exact,loo_cap);
NonL=local_nonlinearity(X,Y,neighbours,model);

A=[CS,eLOOCV,NonL]; B=benefit_normalize(A); pareto_idx=pareto_front(B);
if isempty(pareto_idx), pareto_idx=(1:n)'; end
Bp=B(pareto_idx,:); w=entropy_weights(Bp); Z=Bp.*w;
ideal=max(Z,[],1); nadir=min(Z,[],1);
dp=sqrt(sum((Z-ideal).^2,2)); dm=sqrt(sum((Z-nadir).^2,2));
score=dm./max(dp+dm,eps); [~,ib]=max(score); selected_cell=pareto_idx(ib);

cell_candidates=find(cell_id==selected_cell & isfinite(dist_to_sample));
if isempty(cell_candidates)
    [~,k]=min(pdist2(Xcand,X(selected_cell,:))); cell_candidates=k;
end
Xcell=mc(cell_candidates,:);
if size(mc,1)~=size(Xcand,1) || any(any(mc~=Xcand))
    [~,map_idx]=min(pdist2(Xcand,Xcell),[],1); Xcell_driver=Xcand(unique(map_idx),:);
else
    Xcell_driver=Xcell;
end
[~,s_pred,model.MMRGP]=predict_resp(model.MMRGP,Xcell_driver);
var_eff=predictive_variance(s_pred,m,model);
[~,kbest]=max(var_eff); x_star=Xcell_driver(kbest,:);

[~,idx]=min(pdist2(Xcand,x_star));
if isfield(model,'candidate0') && ~isempty(model.candidate0)
    [tf,idx0]=ismember(x_star,model.candidate0,'rows'); if tf, idx=idx0; end
end
iter_idx=size(model.now_x,1)-size(model.init_x,1)+1;
if ~isfinite(iter_idx) || iter_idx<1, iter_idx=1; end
if ~isfield(model,'M_ind') || numel(model.M_ind)<iter_idx, model.M_ind(iter_idx)=0; end
model.M_ind(iter_idx)=idx;
model.C10=struct('CS',CS,'eLOOCV',eLOOCV,'NonL',NonL,'decision_matrix',A,...
    'normalized_matrix',B,'pareto_idx',pareto_idx,'entropy_weights',w,...
    'topsis_score',score,'selected_cell',selected_cell,'cell_id',cell_id,...
    'mc_points',mc,'loo_exact',loo_exact,'loo_cap',loo_cap);
end

function [id,dmin]=nearest_cell(P,X)
id=zeros(size(P,1),1); dmin=zeros(size(P,1),1); block=50000;
for first=1:block:size(P,1)
    last=min(first+block-1,size(P,1)); D=pdist2(P(first:last,:),X);
    [dmin(first:last),id(first:last)]=min(D,[],2);
end
end

function N=delaunay_neighbours(X)
n=size(X,1); d=size(X,2); N=cell(n,1);
if d==1
    [~,ord]=sort(X(:,1)); for k=1:n, if k>1,N{ord(k)}(end+1)=ord(k-1);end,if k<n,N{ord(k)}(end+1)=ord(k+1);end,end
    return
end
try
    S=delaunayn(X);
    for i=1:size(S,1), v=S(i,:); for a=1:numel(v), for b=a+1:numel(v), N{v(a)}(end+1)=v(b);N{v(b)}(end+1)=v(a);end,end,end
catch
    k=min(max(d+1,2),n-1); D=pdist2(X,X); D(1:n+1:end)=inf; [~,ix]=sort(D,2,'ascend'); for i=1:n,N{i}=ix(i,1:k);end
end
for i=1:n
    N{i}=unique(N{i}(N{i}~=i));
    if isempty(N{i}) && n>1, jj=[1:i-1,i+1:n]; [~,q]=min(pdist2(X(i,:),X(jj,:))); N{i}=jj(q); end
end
end

function e=loo_errors(model,X,Y,exact,cap)
n=size(X,1); m=size(Y,2); e=zeros(n,1); cp=get_classpos(model,m);
scale=max(max(Y,[],1)-min(Y,[],1),std(Y,0,1)); scale(scale<=eps)=1;
for i=1:n
    use_exact=exact && i<=cap;
    if use_exact
        try
            mm=MMRGP0(Y([1:i-1,i+1:n],:),X([1:i-1,i+1:n],:),cp,size(X,2));
            [yp,~,~]=predict_resp(mm,X(i,:));
        catch
            use_exact=false;
        end
    end
    if ~use_exact
        try [yp,~,model.MMRGP]=predict_resp(model.MMRGP,X(i,:)); catch, yp=Y(i,:); end
    end
    er=zeros(1,m);
    for j=1:m
        if cp(j)==2, er(j)=double(round(double(yp(j)))~=round(double(Y(i,j))));
        else, er(j)=abs(double(yp(j))-double(Y(i,j)))/scale(j); end
    end
    e(i)=mean(er);
end
end

function nl=local_nonlinearity(X,Y,N,model)
n=size(X,1); m=size(Y,2); cp=get_classpos(model,m); scale=max(max(Y,[],1)-min(Y,[],1),std(Y,0,1)); scale(scale<=eps)=1; nl=zeros(n,1);
for i=1:n
    ids=unique([i,N{i}]); XX=[X(ids,:),ones(numel(ids),1)]; rr=zeros(numel(ids),m);
    for j=1:m
        beta=pinv(XX)*Y(ids,j); rr(:,j)=abs(XX*beta-Y(ids,j)); if cp(j)~=2,rr(:,j)=rr(:,j)/scale(j);end
    end
    nl(i)=mean(rr(:));
end
end

function cp=get_classpos(model,m)
if isfield(model,'ClassPos'),cp=model.ClassPos;elseif isfield(model,'MMRGP')&&isfield(model.MMRGP,'classpos'),cp=model.MMRGP.classpos;else,cp=ones(1,m);end
cp=cp(:)';cp(end+1:m)=1;cp=cp(1:m);
end

function V=predictive_variance(s,m,model)
n=size(s,1)/m; V=zeros(n,1); W=ones(1,m)/m;
if isfield(model,'c10_response_weights')&&numel(model.c10_response_weights)==m,W=model.c10_response_weights(:)';W=W/sum(W);end
for j=1:m
    vv=max(0,s(j:m:end,j)); vv=vv/(max(vv)+eps); V=V+W(j)*vv;
end
end

function B=benefit_normalize(A)
mn=min(A,[],1);mx=max(A,[],1);B=(A-mn)./max(mx-mn,eps);B(:,mx-mn<=eps)=1;
end

function idx=pareto_front(B)
n=size(B,1);keep=true(n,1);
for i=1:n
    dom=all(B>=B(i,:),2)&any(B>B(i,:),2);dom(i)=false;
    if any(dom),keep(i)=false;end
end
idx=find(keep);
end

function w=entropy_weights(B)
if size(B,1)<=1,w=ones(1,size(B,2))/size(B,2);return;end
P=B./max(sum(B,1),eps);P(P<=0)=eps;k=1/log(size(B,1));en=-k*sum(P.*log(P),1);q=1-en;
if sum(q)<=eps,w=ones(1,size(B,2))/size(B,2);else,w=q/sum(q);end
end
