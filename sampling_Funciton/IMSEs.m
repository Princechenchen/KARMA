function [x_star, model] =  IMSEs(model)   
ns = size(model.candidate,1); yd = size(model.init_value,2); nts = size(model.testsample,1);
canddata = model.candidate; %原始候选集备份
% Im = ones(model.MMRGP.m,1);  
Im = eye(size(model.ClassPos,2));         % n×n的单位矩阵
cov_TX=define_covmatrix0(model.MMRGP,model.testsample,model.MMRGP.X);  %cov:testx_X
testx = model.testsample(1,:);
cov_tt = define_covmatrix0(model.MMRGP,testx,testx);  %cov:testx_testx
for i=1:ns
    i 
    candx = model.candidate(i,:);
    cov_ct=define_covmatrix0(model.MMRGP,candx,model.testsample);  %cov:candx_testX
    cov_cX=define_covmatrix0(model.MMRGP,candx,model.MMRGP.X);  %cov:candx_X
    cov_cc=define_covmatrix0(model.MMRGP,candx,candx);  %cov:candx_candx
    % new矩阵部件计算
    zmn_new =  [[model.MMRGP.z_mn;cov_cX],[cov_cX';cov_cc]]; %z_(m(n+1)*m(n+1))拼接
    zchol_new=poschol(zmn_new); %chol_SIGMA_new  
    invz_new = zchol_new\(zchol_new'\eye(size(zchol_new))); %inv_SIGMA
    F_new = [model.MMRGP.fn;Im];   %F(n+1)
    SigmaF_new=(zchol_new\(zchol_new'\F_new)); %inv_SIGMA_new*F(n+1)
    FSigmaF_new=F_new'*SigmaF_new;  %F(n+1)'*inv_SIGMA_new*F(n+1)
    FSigmaF_chol2=poschol(FSigmaF_new);  %chol(F(n+1)'*inv_SIGMA_new*F(n+1)
    invFSigmaF_new = FSigmaF_chol2\(FSigmaF_chol2'\eye(size(FSigmaF_chol2)));
    for j=1:nts
        % uncertainty of predict var
%         testx = model.testsample(j,:);
        cov_tX = cov_TX(1+yd*(j-1):yd*j,:);
%         cov_tX=define_covmatrix0(model.MMRGP,testx,model.MMRGP.X);  %cov:testx_X
        cov_tXc = [cov_tX,cov_ct(:,1+yd*(j-1):yd*j)];        %cov:testx_(X,candx)
%         cov_tt = define_covmatrix0(model.MMRGP,testx,testx)  %cov:testx_testx
%         sigma_new = cov_tt - cov_tXc*invz_new*cov_tXc';    %covtt-covt(X,candx)*cov(X,candx)(X,candx)*covt(X,candx)'
        sigma_new = diag(cov_tt) - sum(cov_tXc*invz_new.*cov_tXc,2);
        % uncertainty of predict mean
        % new
        cov_tXcandx=[cov_tX,cov_ct(:,1+yd*(j-1):yd*j)]; %cov:testx_X+candx 拼接
        u2=Im-cov_tXcandx*SigmaF_new; % u2=Im-cov(testx,x+candx)*inv_SIGMA_new*F(n+1)
%         uFSigmaFu_new=u2*(FSigmaF_chol2\(FSigmaF_chol2'\u2'));  % u2*inv(F(n+1)*inv_SIGMA_new*F(n+1))*u2'
        uFSigmaFu_new=sum(u2*invFSigmaF_new.*u2,2);
        
        sigma_new = sigma_new+uFSigmaFu_new;
        varV(:,j) = sigma_new;
%         varV(:,j) = diag(sigma_new); %??? 是否加和，是否只取对角线
    end
%     SUR_metric(:,:,i)=delta_uncertainty;
%     mdvar(:,i) = sum(dvar,2);
%     metric(i) = sum(diag(delta_sigma));
    % none weight
    metric(i) = min(sum(varV,2)./(max(varV,[],2)-min(varV,[],2)));
    
end
% distvar = dist(model.candidate,model.MMRGP.X'); %distance_weight
% distvar = min(distvar,[],2);

% distance weight
% metric = metric.*distvar';

% %   weight
% meandvar = mean(mdvar,2); maxdvar=max(mdvar,[],2);
% weight = 0.5*(meandvar./sum(meandvar)+maxdvar./sum(maxdvar));
% metric =  distvar'.*sum(mdvar.*weight); 

[bestvalue,index] = min(metric);
x_star = model.candidate(index,:);
[index] = ismember(canddata, x_star, 'rows');  %返回最佳点在原始候选集中的索引
index = find(index);
[index0] = ismember(model.candidate0, x_star, 'rows');  %返回最佳点在原始候选集中的索引
index0 = find(index0);
model.M_ind(size(model.now_value,1)-size(model.init_value,1)+1)=index0(1);
canddata(index,:)=[];
model.candidate = canddata;
% model.M_ind(size(model.now_value,1)-size(model.init_value,1)+1)=index;  %ship案例需要这个，且需要注释上面的7行%%