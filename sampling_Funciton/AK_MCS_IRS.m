function [x_star, model] =  AK_MCS_IRS(model,obj_fct)
ns = size(model.candidate,1); yd = size(model.init_value,2);
canddata = model.candidate; %原始候选集备份
% [y_hat_candidate, s, model.MMRGP]=predict_resp(model.MMRGP,model.candidate);
% for i=1:yd
%     var(:,i) = s(i:yd:end,i);
% end
xd = size(model.candidate,2);
mu = 0.5*ones(1,xd);        % 均值（X1=1.5, X2=2.5）
sigma = 0.166*ones(1,xd);         % 标准差（X1=1, X2=1）3sigma
d = 1/4;                % 冗余区域距离因子（论文推荐1/4）
IF = 0.8;               % 重要性因子（一般情况0.8，特殊情况1.0）
% 初始化
if size(model.init_x,1)==size(model.now_x,1)
    %% 2. 步骤1：生成初始DoE（关键点法，替代拉丁超立方采样）
    % 2.1 生成2^n个边界点（每个维度取mu±3*sigma）
    bound_pts = [];
    bits = dec2bin(0:2^xd-1, xd) - '0';  % 生成0/1组合（nDim位二进制）
%     % 生成随机索引（无重复） %工程案例专属，因为计算黑箱函数实在是太慢了
%     random_idx = randperm(size(bits, 1), 2*xd);
%     bits = bits(random_idx, :);

    dim_vals = [mu(1)-3*sigma(1), mu(1)+3*sigma(1)];
    bound_pts = dim_vals(1) + bits .* (dim_vals(2) - dim_vals(1));  % 映射到上下界区间
    % 2.2 生成中心点（mu1, mu2, ..., mun）
    center_pt = mu;
    %% 3. 步骤2：定义总区域（TR）和子区域（SR）
    % 3.1 总区域TR：边界点包围的矩形（2维）/立方体（高维）
    TR_lb = mu - 3*sigma;  % TR下界
    TR_ub = mu + 3*sigma;  % TR上界
    % 3.2 子区域SR：TR被中心点划分为2^n个小区域（如2维：4个SR）
    % SR划分规则：每个维度以中心点为界，分为“小于mu”和“大于mu”两类
    SR_list = cell(1, 2^xd);
    for i = 1:2^xd
        % 生成每个SR的二进制编码（如2维：00→X1<mu1,X2<mu2；01→X1<mu1,X2>mu2...）
        code = dec2bin(i-1, xd) - '0';
        SR_lb = TR_lb;
        SR_ub = TR_ub;
        for j = 1:xd
            if code(j) == 0
                SR_ub(j) = mu(j);  % 第j维上界=mu(j)
            else
                SR_lb(j) = mu(j);  % 第j维下界=mu(j)
            end
        end
        SR_list{i} = [SR_lb; SR_ub];  % 存储每个SR的[下界; 上界]
    end
    %% 5. 步骤4：动态重要区域（IR）/不确定区域（UR）初始化
    % 核心规则：若中心点LSF与某边界点LSF符号相反，则该边界点所在SR为IR
    center_G = obj_fct(center_pt);  center_G = center_G(model.ClassPos==2);% 中心点LSF值
%     bound_G = obj_fct(bound_pts); %一般情况用这个
    load('bound_G.mat'); %tenbar案例专属
    bound_G = bound_G(:,model.ClassPos==2);% 边界点LSF值
    IR_idx = [];                % IR对应的SR索引
    UR_idx = [];                % UR对应的SR索引
    for i = 1:size(bound_pts,1)
        % 找到当前边界点所在的SR（通过判断是否在SR的[下界,上界]内）
        bound_pt = bound_pts(i,:);
        for sr = 1:2^xd
            sr_lb = SR_list{sr}(1,:);
            sr_ub = SR_list{sr}(2,:);
            in_sr = all(bound_pt >= sr_lb - 1e-8) && all(bound_pt <= sr_ub + 1e-8);
            if in_sr
                % 符号相反 → 加入IR；否则加入UR
                if sign(center_G) ~= sign(bound_G(i)) && ~ismember(sr, IR_idx)
                    IR_idx = [IR_idx, sr];
                elseif ~ismember(sr, IR_idx) && ~ismember(sr, UR_idx)
                    UR_idx = [UR_idx, sr];
                end
                break;
            end
        end
    end
    model.SR_list = SR_list;
    model.IR_idx = IR_idx;
    model.UR_idx = UR_idx;
end
%% 6. 步骤5：选取更新样本（核心循环）
SR_list = model.SR_list;
IR_idx = model.IR_idx;
UR_idx = model.UR_idx;
% 6.1 生成总区域（TR）样本（用于筛选新样本点）
X_TR = model.candidate;

% 6.2 按重要性因子IF划分IR/UR样本池
% 筛选IR中的样本
X_IR = [];
for sr = IR_idx
    sr_lb = SR_list{sr}(1,:);
    sr_ub = SR_list{sr}(2,:);
    in_IR = all(X_TR >= sr_lb - 1e-8, 2) & all(X_TR <= sr_ub + 1e-8, 2);
    X_IR = [X_IR; X_TR(in_IR,:)];
end
% 筛选UR中的样本
X_UR = [];
for sr = UR_idx
    sr_lb = SR_list{sr}(1,:);
    sr_ub = SR_list{sr}(2,:);
    in_UR = all(X_TR >= sr_lb - 1e-8, 2) & all(X_TR <= sr_ub + 1e-8, 2);
    X_UR = [X_UR; X_TR(in_UR,:)];
end

% 6.3 按IF概率选择样本池（IR占IF，UR占1-IF）
N_IR_select = round(IF * ns);
N_UR_select = ns - N_IR_select;
% 避免IR/UR样本不足的情况
N_IR_select = min(N_IR_select, size(X_IR,1));
N_UR_select = min(N_UR_select, size(X_UR,1));
% 随机抽取样本
if isempty(X_IR)
    idx_UR = randi(size(X_UR,1), size(X_UR,1), 1);
    X_candidate = X_UR(idx_UR,:);
elseif isempty(X_UR)
    idx_IR = randi(size(X_IR,1), N_IR_select, 1);
    X_candidate = X_IR(idx_IR,:);
else
    idx_UR = randi(size(X_UR,1), N_UR_select, 1);
    idx_IR = randi(size(X_IR,1), N_IR_select, 1);
    X_candidate = [X_IR(idx_IR,:); X_UR(idx_UR,:)];
end
% idx_IR = randi(size(X_IR,1), N_IR_select, 1);
% if isempty(X_UR)
%     X_candidate = X_IR(idx_IR,:);
% else
%     idx_UR = randi(size(X_UR,1), N_UR_select, 1);
%     X_candidate = [X_IR(idx_IR,:); X_UR(idx_UR,:)];
% end
ns = size(X_candidate,1); 

% 6.4 用学习函数U筛选最优样本点（U最小=不确定性最大）  %%%%这里分两个版本，1.U函数版，2.只用关键区域方法结合SUR
[y, s, model.MMRGP]=predict_resp(model.MMRGP,X_candidate);
for i=1:yd
    var(:,i) = s(i:yd:end,i);
end
yc = model.MMRGP.fc;
varc = var(:,model.ClassPos==2);
U = abs(yc)./varc;
[bestvalue,index] = min(U);
X_new = X_candidate(index,:);                     % 新样本点
G_new = obj_fct(X_new); G_new = G_new(model.ClassPos==2);  % 计算真实LSF值（调用真实模型）

% 6.5 冗余区域（RR）排除：删除TR中与X_new过近的样本
RR_lb = X_new - d*sigma;  % RR下界
RR_ub = X_new + d*sigma;  % RR上界
% 更新TR：删除RR内的点（避免后续样本重复）
in_RR = all(X_TR >= RR_lb - 1e-8, 2) & all(X_TR <= RR_ub + 1e-8, 2);
X_TR = X_TR(~in_RR,:);
model.candidate = X_TR;  %传递下一次循环

% 6.9 动态更新IR/UR（若新样本在UR且LSF符号与中心点相反）
% 判断新样本是否在UR
in_UR_flag = false;
new_sr = [];
for sr = UR_idx
    sr_lb = SR_list{sr}(1,:);
    sr_ub = SR_list{sr}(2,:);
    if all(X_new >= sr_lb - 1e-8) && all(X_new <= sr_ub + 1e-8)
        in_UR_flag = true;
        new_sr = sr;
        break;
    end
end
center_G = obj_fct(mu); center_G = center_G(model.ClassPos==2);  % 计算中心点标签（调用真实模型）
% 符号相反 → 将该SR加入IR，从UR删除
if in_UR_flag && sign(G_new) ~= sign(center_G)
    if ~ismember(new_sr, IR_idx)
        IR_idx = [IR_idx, new_sr];
        UR_idx = UR_idx(UR_idx ~= new_sr);  % 从UR中删除
    end
end
model.IR_idx = IR_idx;  %传递下一次循环
model.UR_idx = UR_idx;

% 步骤5：选择“信息增益/旅行成本”最大的点
x_star = X_new;
[index] = ismember(canddata, x_star, 'rows');  %返回最佳点在原始候选集中的索引
index = find(index);
[index0] = ismember(model.candidate0, x_star, 'rows');  %返回最佳点在原始候选集中的索引
index0 = find(index0);
model.M_ind(size(model.now_value,1)-size(model.init_value,1)+1)=index0;
% canddata(index,:)=[];
% model.candidate = canddata;
