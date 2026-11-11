function Y_pred = Y(A, model,model_num)


%Y_pred=SVRPredict(A, model);
    % %A的输入格式为A=[0,0,0,0;1,1,1,1]
    % if model_num==0
    %     %Y_pred = test2(A, model);
         [Y_pred, s, modelnew]=predict_resp(model.MMRGP,A);
         Y_pred(:,model.ClassPos==2) = modelnew.fc;
    % elseif model_num==1
    %     % [~,Y_pred] = BSVRmodelPredict(A, model);
    %     Y_pred=SVRPredict(A, model);
    % elseif model_num==2
    %     [~,Y_pred] = LSSVRmodelPredict(A, model);
    % end
end