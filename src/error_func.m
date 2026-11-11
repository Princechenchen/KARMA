function error_value = error_func(vtrue,vpred,ClassPos)
v_true = vtrue(:,find(ClassPos==1)); v_pred =vpred(:,find(ClassPos==1));
RMAE = mean(abs(v_true-v_pred));
RMSE = sqrt(mean(((v_true-v_pred).^2)));
MAPE = mean(abs((v_true-v_pred)./v_true));
SS_res = sum((v_true - v_pred).^2);
SS_tot = sum((v_true - mean(v_true)).^2);
R2 = 1 - SS_res ./ SS_tot;
index = find(ClassPos==2);
for i=1:length(index)
v_true = vtrue(:,index(i)); v_pred =vpred(:,index(i));
ACCURACY(i) = sum(v_pred==v_true)/size(v_true,1);
[A,~] = confusionmat(v_true,v_pred);
if size(A,1)>1
c1_precise = A(2,2)/(A(1,2) + A(2,2));
c1_recall = A(2,2)/(A(2,1) + A(2,2));
F1(i) = 2 * c1_precise * c1_recall/(c1_precise + c1_recall);
else
F1(i) = 1;
end
end
error_value={RMAE,RMSE,MAPE,ACCURACY,R2};
end
