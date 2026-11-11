clear; close all;
[parent_dir,~,~] = fileparts(pwd);
addpath([parent_dir,'/src']); addpath([parent_dir,'/sampling_Funciton']);
addpath([parent_dir,'/testFunc']); addpath([parent_dir,'\grid']);
addpath([parent_dir,'\MRSM_subprograms1']); addpath([parent_dir,'\MS-PSO-pro']); addpath(parent_dir);
WARMUP = 5;
resume_if_checkpoint = true;
progf = fullfile(pwd,'branin_box3d_prog.txt');
fp = fopen(progf,'w'); fprintf(fp,'branin box3d start %s (warmup=%d)\n', datetime('now'), WARMUP); fclose(fp);
SUBSET_CELLS = [2,3,4,6,7,8,9,10,11,14];
rec = load(fullfile(pwd,'rec0407Branin_Gbest.mat'), 'modelmeanMSE');
b4e = load(fullfile(pwd,'rec_branin_ext4_base.mat'));  b4e = b4e.base4;
b42 = load(fullfile(pwd,'rec_branin_ext2_base.mat'));  b42 = b42.base4;
nC = numel(SUBSET_CELLS);
bases = cell(1,nC);  seeds = zeros(1,nC);
for k = 1:nC
    c = SUBSET_CELLS(k);
    if c <= 10
        m = rec.modelmeanMSE{c};
        bases{k} = struct('rs',m.rs,'candidate0',m.candidate0,'init_x',m.init_x, ...
            'init_value',m.init_value,'testset',m.testset,'testsample',m.testsample);
    elseif c <= 14
        b = b4e(c-10);
        bases{k} = struct('rs',b.rs,'candidate0',b.candidate0,'init_x',b.init_x, ...
            'init_value',b.init_value,'testset',b.testset,'testsample',b.testsample);
    else
        b = b42(c-14);
        bases{k} = struct('rs',b.rs,'candidate0',b.candidate0,'init_x',b.init_x, ...
            'init_value',b.init_value,'testset',b.testset,'testsample',b.testsample);
    end
    seeds(k) = bases{k}.rs;
end
fp = fopen(progf,'a'); fprintf(fp,'cells=[%s] seeds=[%s]\n', num2str(SUBSET_CELLS), num2str(seeds)); fclose(fp);
arm_names = {'C8uc-box3d-COMP'};
arm_samplers = {@mROI_SUR2w_varcs02lPD_fast_PI_pil3brd};
nA = numel(arm_names);
obj_fct = @(x)Branin_G(x);
base = struct();
base.variables.ns = 100; base.variables.dim = 2; base.dim = 2;
base.ClassPos = [2,1,1];
base.total_iter = 50; base.n_init = 30;
ccols = find(base.ClassPos==1);  nRC = numel(ccols);
resp_lab = {'y1 = 2*ell-3','y2 = ell^2'};
bounds01 = [0 1; 0 1];
hg_centers = high_gradient_centers(@Branin_2d3, bounds01, 2, [], [], 'relcurv');
hw = 0.25;
boxes = [hg_centers(:,1)-hw, hg_centers(:,1)+hw, hg_centers(:,2)-hw, hg_centers(:,2)+hw];
boxes = max(min(boxes,1),0);
P1 = [0.284, 0.828]; P2 = [0.746, 0.441]; w = 0.04;
dvec = P2 - P1; dvec = dvec/norm(dvec);
nvec = [-dvec(2), dvec(1)];
ribbon = [P1 + w*nvec; P2 + w*nvec; P2 - w*nvec; P1 - w*nvec];
[pi_hg_region,~]  = construct_pi_phys('region', struct('boxes',boxes,'edge_width',0.06));
[pi_bnd_ribbon,~] = construct_pi_phys('region', struct('regions',{{ribbon}}, 'edge_width',0.06));
arm_comp = cell(1, nA);
arm_comp{1} = {pi_hg_region, pi_bnd_ribbon};
fprintf('===== [branin-box3d] 先验几何(V1 定版 w=0.04) warmup=%d =====\n', WARMUP);
fp = fopen(progf,'a'); fprintf(fp,'prior geometry done\n'); fclose(fp);
n_seq = base.total_iter - base.n_init;
rmse_resp = nan(n_seq, nA, nC, nRC);  rmae_resp = nan(n_seq, nA, nC, nRC);
acc_arr = nan(n_seq, nA, nC);  t_samp = nan(n_seq, nA, nC);
alpha_hist = cell(nA, nC);  cons_hist = cell(nA, nC);  picks = cell(nA, nC);
gate_hist = cell(nA, nC);
arm_failed = false(nA, nC);  arm_errmsg = cell(nA, nC);
ckpt = fullfile(pwd,'unified_branin_box3d_checkpoint.mat');
start_ci = 1;
if resume_if_checkpoint && isfile(ckpt)
    C = load(ckpt);
    needf = {'rmse_resp','rmae_resp','acc_arr','t_samp','alpha_hist','cons_hist', ...
             'picks','gate_hist','arm_failed','arm_errmsg','seeds'};
    if all(isfield(C,needf)) && numel(C.seeds)<=nC && isequal(C.seeds,seeds(1:numel(C.seeds))) ...
            && isequal(size(C.rmse_resp,2),nA) && isequal(size(C.rmse_resp,4),nRC)
        k = numel(C.seeds);
        doneCi = find(~all(isnan(reshape(C.acc_arr(end,:,:), nA, k)), 1));
        if ~isempty(doneCi)
            rmse_resp(:,:,1:k,:) = C.rmse_resp;  rmae_resp(:,:,1:k,:) = C.rmae_resp;
            acc_arr(:,:,1:k) = C.acc_arr;  t_samp(:,:,1:k) = C.t_samp;
            alpha_hist(:,1:k) = C.alpha_hist;  cons_hist(:,1:k) = C.cons_hist;
            picks(:,1:k) = C.picks;  gate_hist(:,1:k) = C.gate_hist;
            arm_failed(:,1:k) = C.arm_failed; arm_errmsg(:,1:k) = C.arm_errmsg;
            start_ci = max(doneCi) + 1;
            if start_ci <= nC
                fprintf('续跑: cells 1-%d 已完成,从 %d (cell %d, rs=%d) 开始\n', ...
                    max(doneCi), start_ci, SUBSET_CELLS(start_ci), seeds(start_ci));
            else
                fprintf('续跑: 全部重复已完成,直接汇总\n');
            end
        end
    end
end
for ci = start_ci:nC
    b = bases{ci};
    model = base;
    model.rs         = b.rs;
    model.candidate  = b.candidate0;
    model.candidate0 = b.candidate0;
    model.testsample = b.testsample;
    model.testset    = b.testset;
    model.init_x     = b.init_x;
    model.init_value = b.init_value;
    ytest_fixed      = obj_fct(model.testset);
    model.MMRGP      = MMRGP0(model.init_value, model.init_x, model.ClassPos, model.dim);
    model.now_value  = model.init_value; model.now_x = model.init_x;
    rng(b.rs*100 + SUBSET_CELLS(ci));  rng_state = rng;
    fp = fopen(progf,'a'); fprintf(fp,'cell %d (rs=%d) base ready, init check max|d|=%.2e\n', ...
        SUBSET_CELLS(ci), b.rs, max(abs(obj_fct(b.init_x) - b.init_value),[],'all')); fclose(fp);
    for ai = 1:nA
        if arm_failed(ai, ci), continue; end
        rng(rng_state);
        ma = model;
        ma.pi_comp_fns      = arm_comp{ai};
        ma.pi_comp_adaptive = true;
        ma.pi_comp_eta      = 0.5;
        ma.pi_comp_alpha_min = 0.1;
        ma.pi_comp_alpha    = [1 1];
        ma.pi_comp_alpha_hist = zeros(0,2);
        ma.pi_comp_cons_hist  = zeros(0,2);
        ma.pi3_warmup       = WARMUP;
        fp = fopen(progf,'a'); fprintf(fp,'arm %s start (cell %d)\n', arm_names{ai}, SUBSET_CELLS(ci)); fclose(fp);
        t_arm = tic;
        try
            for i = 1:n_seq
                tt = tic;
                [x_star, ma] = arm_samplers{ai}(ma);
                t_samp(i, ai, ci) = toc(tt);
                f_star = obj_fct(x_star);
                ma.now_value = [ma.now_value; f_star]; ma.now_x = [ma.now_x; x_star];
                ma.MMRGP = MMRGP0(ma.now_value, ma.now_x, ma.ClassPos, ma.dim);
                [ypred] = predict_resp(ma.MMRGP, ma.testset);
                ev = error_func(ytest_fixed, ypred, ma.ClassPos);
                for j = 1:nRC
                    rmae_resp(i, ai, ci, j) = ev{1}(j);
                    rmse_resp(i, ai, ci, j) = sqrt(mean((ytest_fixed(:,ccols(j)) - ypred(:,ccols(j))).^2));
                end
                acc_arr(i, ai, ci)  = mean(ev{4});
                picks{ai, ci}(i,:)  = x_star;
                if mod(i,5)==0 || i==1
                    fprintf('[%s/c%d] iter %d pick=[%.3f %.3f] rmae_y=[%.4f %.4f] acc=%.4f\n', ...
                        arm_names{ai}, SUBSET_CELLS(ci), i, x_star(1), x_star(2), ...
                        rmae_resp(i,ai,ci,1), rmae_resp(i,ai,ci,2), acc_arr(i,ai,ci));
                end
                fp = fopen(progf,'a'); fprintf(fp, '%s cell %d step %d ok (%.1f s)\n', ...
                    arm_names{ai}, SUBSET_CELLS(ci), i, t_samp(i,ai,ci)); fclose(fp);
            end
            if isfield(ma,'pi_comp_alpha_hist'), alpha_hist{ai,ci} = ma.pi_comp_alpha_hist; end
            if isfield(ma,'pi_comp_cons_hist'),  cons_hist{ai,ci}  = ma.pi_comp_cons_hist;  end
            if isfield(ma,'pi3_hist'), gate_hist{ai,ci} = ma.pi3_hist; end
            fprintf('[%s/c%d] 整臂耗时 %.1f min (采样器均值 %.1f s/步)\n', ...
                arm_names{ai}, SUBSET_CELLS(ci), toc(t_arm)/60, ...
                sum(t_samp(:,ai,ci),'omitnan')/sum(~isnan(t_samp(:,ai,ci))));
            fp = fopen(progf,'a'); fprintf(fp,'arm %s done (cell %d)\n', arm_names{ai}, SUBSET_CELLS(ci)); fclose(fp);
        catch err
            arm_failed(ai, ci) = true;
            arm_errmsg{ai, ci} = sprintf('%s | %s', err.identifier, err.message);
            fprintf('[%s/c%d] ★失败: %s\n', arm_names{ai}, SUBSET_CELLS(ci), arm_errmsg{ai,ci});
            fp = fopen(progf,'a'); fprintf(fp,'arm %s FAILED (cell %d): %s\n', ...
                arm_names{ai}, SUBSET_CELLS(ci), err.message); fclose(fp);
        end
        save(ckpt, 'rmse_resp','rmae_resp','acc_arr','t_samp','alpha_hist','cons_hist', ...
             'picks','gate_hist','arm_failed','arm_errmsg','arm_names','seeds','n_seq','resp_lab', ...
             'SUBSET_CELLS','WARMUP','-v7');
    end
    fprintf('=== cell %d/%d (rs=%d) done ===\n', ci, nC, b.rs);
    fp = fopen(progf,'a'); fprintf(fp,'cell %d done\n', SUBSET_CELLS(ci)); fclose(fp);
end
ok = ~arm_failed & ~isnan(reshape(acc_arr(end,:,:), nA, nC));
for j = 1:nRC
    fprintf('\n===== 连续响应 %s 末步 RMAE 均值±std + AULC, cells=[%s] =====\n', ...
        resp_lab{j}, mat2str(SUBSET_CELLS));
    for ai = 1:nA
        if ~any(ok(ai,:))
            fprintf('%-22s ★失败/无数据: %s\n', arm_names{ai}, strjoin(arm_errmsg(ai,:),' ; '));
            continue;
        end
        r = squeeze(rmae_resp(:,ai,:,j));  r = r(:,ok(ai,:));
        fprintf('%-22s %6.4f±%.4f  AULC %8.4f  (%.1f s/步)\n', arm_names{ai}, ...
            mean(r(end,:)), std(r(end,:)), mean(mean(r,1)), ...
            mean(mean(t_samp(:,ai,:),1,'omitnan')));
    end
end
fprintf('\n===== 分类响应 ACC 末步 均值±std =====\n');
for ai = 1:nA
    if ~any(ok(ai,:)), continue; end
    a = squeeze(acc_arr(:,ai,:));  a = a(:,ok(ai,:));
    fprintf('%-22s %6.4f±%.4f\n', arm_names{ai}, mean(a(end,:)), std(a(end,:)));
end
gh = gate_hist{1,1};
if ~isempty(gh)
    fprintf('\ngate 轨迹(cell %s): %s\n', mat2str(SUBSET_CELLS(1)), mat2str(gh(:,4).'));
end
fprintf('\n主循环完成: checkpoint=%s\n 对照分析用 unified_branin_pool16.mat 同 cell 轨迹配对\n', ckpt);
fp = fopen(progf,'a'); fprintf(fp,'ALLDONE\n'); fclose(fp);
exit(0);
