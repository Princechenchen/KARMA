function error_value = error_func(vtrue,vpred,ClassPos)
%% 连续响应
v_true = vtrue(:,find(ClassPos==1)); v_pred =vpred(:,find(ClassPos==1));
% relative mean absolute error RMAE
RMAE = mean(abs(v_true-v_pred)); 
% ./(max(v_true)-min(v_true))
% root mean squared error
RMSE = sqrt(mean(((v_true-v_pred).^2)));
% ./(max(v_true)-min(v_true))
% mean abs percent error
MAPE = mean(abs((v_true-v_pred)./v_true));

%% 分类响应
index = find(ClassPos==2);
for i=1:length(index)
v_true = vtrue(:,index(i)); v_pred =vpred(:,index(i));
% classification accuracy
ACCURACY(i) = sum(v_pred==v_true)/size(v_true,1); %模型预测准确性
% F1-Score
[A,~] = confusionmat(v_true,v_pred);
if size(A,1)>1
c1_precise = A(2,2)/(A(1,2) + A(2,2));
c1_recall = A(2,2)/(A(2,1) + A(2,2));
F1(i) = 2 * c1_precise * c1_recall/(c1_precise + c1_recall);
else
F1(i) = 1;
end

end

%%
error_value={RMAE,RMSE,MAPE,ACCURACY};
end