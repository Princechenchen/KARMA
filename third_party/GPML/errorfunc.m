function errorval = errorfunc(ytest,ypred,delta,n)
    %MSE
    MSE = mse(ypred-ytest);
    RMSE = sqrt(MSE);           %
    RMSEP = RMSE./mean(ytest);
    NRMSE = RMSE./delta;
    %MAE
    MAE = mean(abs(ypred-ytest));
    MAEP = mean(abs(ypred-ytest))/mean(ytest);   %
    NMAE = MAE./delta;
    %R2
    R2 = 1-((sum((ypred-ytest).^2))/sum((ytest-mean(ytest)).^2));
    adjR2 = 1-(1-R2)*(numel(ytest)-1)/(numel(ytest)-1-size(n,2));

    %杂七杂八
    MAPE = mean(abs((ypred-ytest)./ytest));    %
    SMAPE = mean(abs(ypred-ytest)./(0.5*(abs(ypred)+abs(ytest))));
    MSLE = mean(log(abs((1+ytest)./(1+ypred))).^2);
    MedAE = median(abs(ypred-ytest));
    
    errorval = [NRMSE,MAE,MAPE,R2];
end