clear; close all;
[parent_dir,~,~] = fileparts(pwd);
addpath([parent_dir,'/src']); addpath([parent_dir,'/sampling_Funciton']);
addpath([parent_dir,'/testFunc']); addpath([parent_dir,'\grid']);
addpath([parent_dir,'\MRSM_subprograms1']); addpath([parent_dir,'\MS-PSO-pro']); addpath(parent_dir);
run_arm_box3 = true;
pi_unif_grid   = 4;
resume_if_checkpoint = true;
cfg = struct('pi_unif_grid', pi_unif_grid);
rec_file = fullfile(pwd,'rec0407Ackley_best.mat');
progf = fullfile(pwd,'ackley_box3dim_prog.txt');
fp = fopen(progf,'w'); fprintf(fp,'run start %s\n', datetime('now')); fclose(fp);
tmp = load(rec_file, 'modelmemeroy'); rec_memo = tmp.modelmemeroy;
fp = fopen(progf,'a'); fprintf(fp,'loaded modelmemeroy\n'); fclose(fp);
rs_all = cellfun(@(m) m.rs, rec_memo);
rep_cells = 1:10;
seeds = rs_all(rep_cells);
nS = numel(seeds);
only_rep = 0;
orf = fullfile(pwd,'box3dim_only_rep.txt');
if isfile(orf)
    only_rep = round(str2double(strtrim(fileread(orf))));
    if ~ismember(only_rep, 1:nS), only_rep = 0; end
    if only_rep > 0
        fprintf('单 rep 隔离模式: 只跑重复 %d (rs=%d); 其余 rep 数据在 checkpoint 原样保留\n', ...
            only_rep, seeds(only_rep));
    end
end
arm_names = {}; arm_comp = {};
if run_arm_box3
    arm_names{end+1} = 'C8uc-box3dim-COMP';    arm_comp{end+1} = [];
end
arm_attach = repmat({'comp'}, 1, numel(arm_names));
arm_samplers = repmat({@mROI_SUR2w_varcs02lPD_fast_PI_pil3_dim}, 1, numel(arm_names));
nA = numel(arm_names);
run_arms = 1:nA;
obj_fct = @(x)ackley(x);
base = struct();
base.variables.ns = 200; base.variables.dim = 6; base.dim = 6;
base.ClassPos = [1,1,2];
base.total_iter = 50; base.n_init = 20;
ccols = find(base.ClassPos==1); nRC = numel(ccols);
resp_lab = {'y1 = ackley_6d','y2 = ackley_6d3'};
assert(numel(resp_lab)==nRC, 'resp_lab 与连续响应数不一致');
bnd_thr = 35;
hw = 0.42; ew = 0.12;
r0 = 0.555; tg = 0.06;
anc3 = [0.50 0.50 0.50 0.00 0.00 0.00;
        0.50 0.00 0.00 1.00 1.00 1.00;
        0.00 1.00 0.50 1.00 1.00 0.00];
pi_hg_box = @(X) priunion(X, anc3, hw, ew);
pi_shell  = @(X) min(max(exp(-(vecnorm(X,2,2)/sqrt(6) - r0).^2/(2*tg^2)), 0.01), 1);
for ai = 1:nA
    arm_comp{ai} = {pi_hg_box, pi_shell};
end
u4 = linspace(0,1,4); U4 = cell(1,6); [U4{:}] = ndgrid(u4);
U4 = cell2mat(cellfun(@(m)m(:),U4,'uni',0));
cov_hg  = mean(pi_hg_box(U4) > 0.5);
cov_bnd = mean(pi_shell(U4)  > 0.5);
fprintf('===== Ackley 6D box3 臂先验几何(与 box 轮相同) + uconf 裁判场 =====\n');
for k = 1:3
    fprintf('锚%d = [%s] ρ=%.3f\n', k, sprintf('%.2f ',anc3(k,:)), vecnorm(anc3(k,:),2,2)/sqrt(6));
end
fprintf('hw=%.2f ew=%.2f | 壳 ρ₀=%.3f τ=%.3f | 4^6 格覆盖: π_hg>0.5 %.3f, π_bnd>0.5 %.3f\n', ...
    hw, ew, r0, tg, cov_hg, cov_bnd);
hb = figure('Name','ackley box3 prior','Position',[60 80 1400 420],'Visible','off');
subplot(1,3,1);
rr = linspace(0,1,400);
plot(rr, exp(-(rr-r0).^2/(2*tg^2)),'b-','LineWidth',1.8); grid on;
xline(r0,'k--','\rho_0=0.555'); xlabel('\rho = ||x||_2/\surd6'); ylabel('\pi_{bnd}');
title(sprintf('薄壳: \\rho_0=%.3f, \\tau=%.3f (同 box)', r0, tg));
subplot(1,3,2);
tcol = linspace(0,1,200)';
paths = {tcol*ones(1,6), anc3(1,:) + tcol.*(anc3(2,:)-anc3(1,:)), ...
         anc3(2,:) + tcol.*(anc3(3,:)-anc3(2,:))};
pl = gobjects(1,3); hold on; grid on;
for q = 1:3, pl(q) = plot(tcol, pi_hg_box(paths{q}),'-','LineWidth',1.6); end
legend(pl, {'主对角线 0→1','锚1→锚2','锚2→锚3'},'Location','best');
xlabel('路径参数 t'); ylabel('\pi_{hg}');
title(sprintf('3 宽软盒 (hw=%.2f, ew=%.2f) 沿域内路径', hw, ew));
subplot(1,3,3); axis off;
text(0.05,0.72, sprintf('锚1 = [%s]  \\rho=0.354', sprintf('%.2f ',anc3(1,:))));
text(0.05,0.50, sprintf('锚2 = [%s]  \\rho=0.736', sprintf('%.2f ',anc3(2,:))));
text(0.05,0.28, sprintf('锚3 = [%s]  \\rho=0.736', sprintf('%.2f ',anc3(3,:))));
text(0.05,0.06, sprintf('唯一差异: cons_{bnd} 靶场 LOO残差\\rightarrow uconf'));
title('box3 = box 几何 + uconf 裁判场');
saveas(hb, fullfile(pwd,'fig_ackley_box3dim_prior.png'));
close all;
fp = fopen(progf,'a'); fprintf(fp,'prior geometry done\n'); fclose(fp);
n_seq = base.total_iter - base.n_init;
rmse_resp = nan(n_seq, nA, nS, nRC);  rmae_resp = nan(n_seq, nA, nS, nRC);
acc_arr = nan(n_seq, nA, nS);
t_samp    = nan(n_seq, nA, nS);
roi_step1  = cell(nA, nS);
alpha_hist = cell(nA, nS);  cons_hist = cell(nA, nS);
picks      = cell(nA, nS);
arm_failed = false(nA, nS);  arm_errmsg = cell(nA, nS);
hist_ci    = zeros(1, nS);
ckpt = fullfile(pwd,'unified_ackley_box3dim_checkpoint.mat');
tmp_ckpt = fullfile(pwd,'unified_ackley_box3dim_ckpt_tmp.mat');
start_si = 1;
if resume_if_checkpoint && isfile(ckpt)
    C = load(ckpt);
    if isfield(C,'rmse_resp') && isfield(C,'rmae_resp') ...
            && numel(C.seeds)<=nS && isequal(C.seeds,seeds(1:numel(C.seeds))) ...
            && isequal(size(C.rmse_resp,2),nA) && isequal(size(C.rmse_resp,4),nRC) ...
            && isequal(C.arm_names, arm_names)
        k = numel(C.seeds);
        doneSi = find(~any(isnan(reshape(C.acc_arr(end,:,:), nA, k)), 1));
        if ~isempty(doneSi)
            rmse_resp(:,:,1:k,:) = C.rmse_resp;  rmae_resp(:,:,1:k,:) = C.rmae_resp;
            acc_arr(:,:,1:k) = C.acc_arr;  t_samp(:,:,1:k) = C.t_samp;
            roi_step1(:,1:k) = C.roi_step1;  alpha_hist(:,1:k) = C.alpha_hist;
            cons_hist(:,1:k) = C.cons_hist;  picks(:,1:k)      = C.picks;
            arm_failed(:,1:k) = C.arm_failed; arm_errmsg(:,1:k) = C.arm_errmsg;
            hist_ci(1:k) = C.hist_ci;
            start_si = max(doneSi) + 1;
            if start_si <= nS
                fprintf('续跑: 重复 1-%d 已完成,从重复 %d (cell %d, rs=%d) 开始\n', ...
                    max(doneSi), start_si, rep_cells(start_si), seeds(start_si));
            else
                fprintf('续跑: 全部重复已完成,直接汇总\n');
            end
        end
    end
end
si_end = nS;
if only_rep > 0
    start_si = only_rep;
    if all(~isnan(reshape(acc_arr(end,:,only_rep),nA,1)) & ~arm_failed(:,only_rep))
        fprintf('重复 %d 已在 checkpoint 完成, 本次跳过\n', only_rep);
        fp = fopen(progf,'a'); fprintf(fp,'rep %d already complete, skip\n', only_rep); fclose(fp);
        start_si = nS + 1;
    end
    si_end = only_rep;
end
for si = start_si:si_end
    rs = seeds(si);
    ci = rep_cells(si);
    hist_ci(si) = ci;
    hj = rec_memo{ci};
    assert(size(hj.candidate0,1)==200 && size(hj.testset,1)==5000 ...
        && size(hj.init_x,1)==20 && size(hj.testsample,1)==12000, '存档基座尺寸异常');
    model = base;
    model.rs         = hj.rs;
    model.candidate  = hj.candidate0;
    model.candidate0 = hj.candidate0;
    model.testsample = hj.testsample;
    model.testset    = hj.testset;
    model.init_x     = hj.init_x;
    model.init_value = hj.init_value;
    ytest_fixed      = obj_fct(model.testset);
    fprintf('[seed r%d-c%d/rs=%d] init_value 重算校验 max|Δ|=%.2e | testset 正类占比=%.4f\n', ...
        si, ci, rs, max(abs(obj_fct(hj.init_x) - hj.init_value),[],'all'), ...
        mean(ytest_fixed(:, find(base.ClassPos==2))));
    model.MMRGP      = MMRGP0_correxp(model.init_value, model.init_x, base.ClassPos, base.dim);
    model.now_value  = model.init_value; model.now_x = model.init_x;
    rng(rs*100 + ci);  rng_state = rng;
    for ai = run_arms
        if arm_failed(ai, si), continue; end
        rng(rng_state);
        ma = model;
        ma = attach_arm(ma, arm_attach{ai}, [], arm_comp{ai}, cfg);
        fp = fopen(progf,'a'); fprintf(fp,'arm %s (r%d) start\n', arm_names{ai}, si); fclose(fp);
        i_start = 1;
        if resume_if_checkpoint && isfile(ckpt)
            C2 = load(ckpt);
            if isfield(C2,'prog_state') && isfield(C2,'rmae_resp') ...
                    && isfield(C2.prog_state,'si') && C2.prog_state.si==si && C2.prog_state.ai==ai ...
                    && C2.prog_state.i < n_seq && isequal(C2.arm_names, arm_names)
                ps = C2.prog_state;
                rmae_resp = C2.rmae_resp; rmse_resp = C2.rmse_resp; acc_arr = C2.acc_arr;
                t_samp = C2.t_samp; roi_step1 = C2.roi_step1;
                alpha_hist = C2.alpha_hist; cons_hist = C2.cons_hist;
                picks = C2.picks; arm_failed = C2.arm_failed; arm_errmsg = C2.arm_errmsg;
                hist_ci = C2.hist_ci;
                ma.now_x = ps.now_x; ma.now_value = ps.now_value;
                ma.MMRGP = MMRGP0_correxp(ma.now_value, ma.now_x, ma.ClassPos, ma.dim);
                if isfield(ma,'pi_comp_alpha')
                    ma.pi_comp_alpha      = ps.pi_comp_alpha;
                    ma.pi_comp_alpha_hist = ps.pi_comp_alpha_hist;
                    ma.pi_comp_cons_hist  = ps.pi_comp_cons_hist;
                end
                rng(ps.rng_state);
                i_start = ps.i + 1;
                fp = fopen(progf,'a'); fprintf(fp,'[resume] r%d %s 从第 %d 步续跑\n', si, arm_names{ai}, i_start); fclose(fp);
                fprintf('[resume] r%d %s 从第 %d 步续跑(α=[%.3f %.3f])\n', si, arm_names{ai}, i_start, ...
                    ma.pi_comp_alpha(1), ma.pi_comp_alpha(2));
            end
        end
        t_arm = tic;
        try
            for i = i_start:n_seq
                tt = tic;
                [x_star, ma] = arm_samplers{ai}(ma);
                t_samp(i, ai, si) = toc(tt);
                f_star = obj_fct(x_star);
                ma.now_value = [ma.now_value; f_star]; ma.now_x = [ma.now_x; x_star];
                ma.MMRGP = MMRGP0_correxp(ma.now_value, ma.now_x, ma.ClassPos, ma.dim);
                [ypred] = predict_resp(ma.MMRGP, ma.testset);
                ev = error_func(ytest_fixed, ypred, ma.ClassPos);
                for j = 1:nRC
                    rmae_resp(i, ai, si, j) = ev{1}(j);
                    rmse_resp(i, ai, si, j) = sqrt(mean((ytest_fixed(:,ccols(j)) - ypred(:,ccols(j))).^2));
                end
                acc_arr(i, ai, si)  = mean(ev{4});
                picks{ai, si}(i,:)  = x_star;
                if i==1 && isfield(ma,'last_roi_table') && ~isempty(ma.last_roi_table)
                    roi_step1{ai,si} = ma.last_roi_table;
                end
                if mod(i,5)==0 || i==1
                    fprintf('[%s/r%d-c%d] iter %d pick=[%.3f %.3f %.3f ...] rmse_y=[%.4f %.4f] acc=%.4f\n', ...
                        arm_names{ai}, si, ci, i, x_star(1), x_star(2), x_star(3), ...
                        rmse_resp(i,ai,si,1), rmse_resp(i,ai,si,2), acc_arr(i,ai,si));
                end
                fp = fopen(progf,'a'); fprintf(fp,'r%d %s step %d ok (%.1f s)\n', ...
                    si, arm_names{ai}, i, t_samp(i,ai,si)); fclose(fp);
                prog_state = struct('si',si,'ai',ai,'i',i,'rng_state',rng, ...
                    'now_x',ma.now_x,'now_value',ma.now_value, ...
                    'pi_comp_alpha',ma.pi_comp_alpha,'pi_comp_alpha_hist',ma.pi_comp_alpha_hist, ...
                    'pi_comp_cons_hist',ma.pi_comp_cons_hist);
                save(tmp_ckpt,'prog_state','rmae_resp','rmse_resp','acc_arr','t_samp', ...
                    'roi_step1','alpha_hist','cons_hist','picks','arm_failed','arm_errmsg', ...
                    'arm_names','seeds','n_seq','resp_lab','hist_ci','-v7');
                movefile(tmp_ckpt, ckpt);
            end
            if isfield(ma,'pi_comp_alpha_hist'), alpha_hist{ai,si} = ma.pi_comp_alpha_hist; end
            if isfield(ma,'pi_comp_cons_hist'),  cons_hist{ai,si}  = ma.pi_comp_cons_hist; end
            fprintf('[%s/r%d-c%d] 整臂耗时 %.1f min (采样器均值 %.1f s/步)\n', ...
                arm_names{ai}, si, ci, toc(t_arm)/60, ...
                sum(t_samp(:,ai,si),'omitnan')/sum(~isnan(t_samp(:,ai,si))));
            fp = fopen(progf,'a'); fprintf(fp,'arm %s (r%d) done\n', arm_names{ai}, si); fclose(fp);
        catch err
            fp = fopen(progf,'a'); fprintf(fp,'arm %s (r%d) FAILED: %s\n', ...
                arm_names{ai}, si, err.message); fclose(fp);
            arm_failed(ai, si) = true;
            arm_errmsg{ai, si} = sprintf('%s | %s', err.identifier, err.message);
            fprintf('[%s/r%d-c%d] ★失败: %s\n', arm_names{ai}, si, ci, arm_errmsg{ai,si});
        end
        save(ckpt, 'rmse_resp','rmae_resp','acc_arr','t_samp','roi_step1','alpha_hist','cons_hist', ...
             'picks','arm_failed','arm_errmsg','arm_names','seeds','n_seq','resp_lab', ...
             'hist_ci','anc3','hw','ew','r0','tg','bnd_thr','-v7');
    end
    fprintf('=== seed %d/%d (rs=%d) done ===\n', si, nS, rs);
end
ok = ~arm_failed & ~isnan(reshape(acc_arr(end,:,:), nA, nS));
for j = 1:nRC
    fprintf('\n===== [box3 臂(uconf 裁判场)] 连续响应 %s 末步 RMAE 均值±std + AULC, seeds=%s =====\n', ...
        resp_lab{j}, mat2str(seeds));
    fprintf('%-20s %14s %12s %10s\n','arm','RMAE','AULC_rmae','s/step');
    for ai = 1:nA
        if ~any(ok(ai,:))
            fprintf('%-20s ★失败/无数据: %s\n', arm_names{ai}, strjoin(arm_errmsg(ai,:),' ; '));
            continue;
        end
        r = squeeze(rmae_resp(:,ai,:,j));  r = r(:,ok(ai,:));
        t = squeeze(t_samp(:,ai,:));       t = t(:,ok(ai,:));
        tmean = mean(t(:));
        if isnan(tmean), tstr = '   (hist)'; else, tstr = sprintf('%8.2f', tmean); end
        fprintf('%-20s %6.4f±%.4f  %8.4f  %s\n', arm_names{ai}, ...
            mean(r(end,:)), std(r(end,:)), mean(mean(r,1)), tstr);
    end
end
fprintf('\n主循环完成: checkpoint=%s\n 运行 post_ackley_box3_merge.m 生成 17 臂总表/图/unified_ackley_box3_results.mat\n', ckpt);
fp = fopen(progf,'a'); fprintf(fp,'ALLDONE\n'); fclose(fp);
function m = attach_arm(m, kind, prior, comp, cfg)
switch kind
    case 'comp'
        if numel(comp)~=2
            error('attach_arm:BadComp','comp 需为 {π_hg, π_bnd} 二元 cell');
        end
        m.pi_comp_fns = comp;
        m.pi_comp_adaptive = true;
        m.pi_comp_eta = 0.5;
        m.pi_comp_alpha_min = 0.1;
        m.pi_comp_alpha = [1 1];
        m.pi_comp_alpha_hist = zeros(0,2);
        m.pi_comp_cons_hist = zeros(0,2);
        m.pi_unif_grid = cfg.pi_unif_grid;
    otherwise
        error('attach_arm:UnknownKind','未知 attach 类型: %s', kind);
end
end
function v = priunion(X, anc, hw, ew)
v = zeros(size(X,1),1);
for c = 1:size(anc,1)
    dout = max(max(anc(c,:)-hw - X, 0), X - (anc(c,:)+hw));
    v = max(v, min(max(exp(-0.5*(max(dout,[],2)/ew).^2), 0.01), 1));
end
end
