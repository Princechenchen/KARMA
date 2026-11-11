% main_hpo_bo_arms —— BO 任务面单臂 × 10 种子小驱动(2026-10-18, ★待用户确认后执行)
% 臂(本文件为模板, 由生成脚本派生三个变体):
%   C14-ROI-EI     = pil3_sob_DS_EI + pi_comp_adaptive=false(静态先验)
%   C14b-ROI-EI-arb = pil3_sob_DS_EI + pi_comp_adaptive=true (仲裁, 与 C8 同款)
%   C15-ROI-Rec    = pil3_sob_DS_REC(交替双通道, 仲裁同 C8)
% 协议: 与主轮逐位可比 —— 同 rec_hpo_base 基座, rng(b.rs*100+cell) 快照 + 逐臂重置,
%   50 步全池查表(断言 bank 命中), 逐臂逐 cell 存档 + 断点续跑(整臂跳过)。
clear; close all;
[parent_dir,~,~] = fileparts(pwd);
addpath([parent_dir,'/src']); addpath([parent_dir,'/sampling_Funciton']);
addpath([parent_dir,'/numeric_sample']);
addpath([parent_dir,'/testFunc']); addpath([parent_dir,'\grid']);
addpath([parent_dir,'\MRSM_subprograms1']); addpath([parent_dir,'\MS-PSO-pro']); addpath(parent_dir);

%% ==== 变体参数(生成时替换) ====
ARM_NAME = 'C15-ROI-Rec';
ARM_FN = @mROI_SUR2w_varcs02lPD_fast_PI_pil3_sob_DS_REC;
ARM_ADAPTIVE = true;      % false=C14 静态; true=C14b/C15 仲裁
CKPT = 'unified_hpo_c15_ckpt.mat';
PROG = 'hpo_bo_c15_prog.txt';
if strcmp(ARM_NAME, 'C15-ROI-Rec'), IS_REC = true; else, IS_REC = false; end

TAU = 0.05;
progf = fullfile(pwd, PROG);
fp = fopen(progf,'w'); fprintf(fp,'bo-arm %s start %s\n', ARM_NAME, datetime('now')); fclose(fp);

BE = load(fullfile(pwd,'rec_hpo_base.mat'));
base10 = BE.base10;
BK = struct('X_bank',double(BE.X_bank), 'y1_bank',double(BE.y1_bank), 'gap_bank',double(BE.gap_bank));
[pi_hg_fn, pi_bnd_fn, ~, PC] = pi_hpo_prior();
assert(abs(BE.TAU - TAU) < 1e-12, 'TAU 不一致');

base = struct();
base.variables.ns = 200; base.variables.dim = 8; base.dim = 8;
base.ClassPos = [1,1,2];
base.total_iter = 70; base.n_init = 20;
ccols = find(base.ClassPos==1); nRC = numel(ccols);
resp_lab = {'y1 = val_acc','y2 = gen_gap'};
n_seq = 50; nS = numel(base10);

rmse_resp = nan(n_seq, 1, nS, nRC); rmae_resp = nan(n_seq, 1, nS, nRC);
acc_arr = nan(n_seq, 1, nS); t_samp = nan(n_seq, 1, nS);
alpha_hist = cell(1, nS); cons_hist = cell(1, nS); picks = cell(1, nS);
channel_log = nan(n_seq, nS);
arm_failed = false(1, nS); arm_errmsg = cell(1, nS);
seeds = arrayfun(@(b) b.rs, base10);

ckpt = fullfile(pwd, CKPT);
if isfile(ckpt)
    C = load(ckpt);
    needf = {'rmae_resp','acc_arr','picks','arm_failed','seeds','t_samp'};
    if all(isfield(C,needf)) && isequal(C.seeds, seeds)
        rmae_resp = C.rmae_resp; rmse_resp = C.rmse_resp; acc_arr = C.acc_arr;
        t_samp = C.t_samp; alpha_hist = C.alpha_hist; cons_hist = C.cons_hist;
        picks = C.picks; channel_log = C.channel_log;
        arm_failed = C.arm_failed; arm_errmsg = C.arm_errmsg;
        fprintf('[resume] 载入 ckpt: %d/%d cells 已完成\n', sum(~isnan(acc_arr(end,1,:))), nS);
    end
end

for k = 1:nS
    if ~isnan(acc_arr(end, 1, k))
        fprintf('[resume] cell %d 已完成, 跳过\n', k);
        continue;
    end
    b = base10(k);
    model = base;
    model.rs = b.rs; model.candidate = b.candidate0; model.candidate0 = b.candidate0;
    model.testsample = b.testsample; model.testset = b.testset;
    model.init_x = b.init_x; model.init_value = b.init_value;
    ytest_fixed = [BK.y1_bank(b.testset_idx), BK.gap_bank(b.testset_idx)];
    ytest_fixed = [ytest_fixed, double(ytest_fixed(:,2) > TAU)];
    model.MMRGP = MMRGP0_correxp(model.init_value, model.init_x, model.ClassPos, model.dim);
    model.now_value = model.init_value; model.now_x = model.init_x;
    rng(b.rs*100 + k); rng_state = rng;
    fp = fopen(progf,'a'); fprintf(fp,'cell %d (rs=%d) start\n', k, b.rs); fclose(fp);
    ma = model;
    ma.pi_comp_fns = {pi_hg_fn, pi_bnd_fn};
    ma.pi_comp_adaptive = ARM_ADAPTIVE;
    ma.pi_comp_eta = 0.5; ma.pi_comp_alpha_min = 0.1; ma.pi_comp_alpha = [1 1];
    ma.pi_comp_alpha_hist = zeros(0,2); ma.pi_comp_cons_hist = zeros(0,2);
    ma.pi_unif_grid = 4; ma.pi_unif_sobol_n = 8192;
    rng(rng_state);
    try
        for i = 1:n_seq
            tt = tic;
            [x_star, ma] = ARM_FN(ma);
            t_samp(i, 1, k) = toc(tt);
            [tf, row] = ismember(x_star, BK.X_bank, 'rows');
            assert(tf, '选点非 bank 行(全池协议违约)');
            y1 = BK.y1_bank(row); gp = BK.gap_bank(row);
            f_star = [y1, gp, double(gp > TAU)];
            ma.now_value = [ma.now_value; f_star]; ma.now_x = [ma.now_x; x_star];
            ma.MMRGP = MMRGP0_correxp(ma.now_value, ma.now_x, ma.ClassPos, ma.dim);
            ypred = predict_resp(ma.MMRGP, ma.testset);
            ev = error_func(ytest_fixed, ypred, ma.ClassPos);
            for j = 1:nRC
                rmae_resp(i,1,k,j) = ev{1}(j);
                rmse_resp(i,1,k,j) = sqrt(mean((ytest_fixed(:,ccols(j)) - ypred(:,ccols(j))).^2));
            end
            acc_arr(i,1,k) = mean(ev{4});
            picks{1, k}(i,:) = x_star;
            if IS_REC, channel_log(i, k) = ma.rec_last_channel; end
            if mod(i,10)==0 || i==1
                fprintf('[%s/c%d] step %d rmae_y1=%.4f acc=%.4f%s\n', ARM_NAME, k, i, ...
                    rmae_resp(i,1,k,1), acc_arr(i,1,k), ...
                    ternary(IS_REC && ma.rec_last_channel==2, ' [rec]', ' [learn]'));
            end
        end
        if isfield(ma,'pi_comp_alpha_hist'), alpha_hist{1,k} = ma.pi_comp_alpha_hist; end
        if isfield(ma,'pi_comp_cons_hist'),  cons_hist{1,k}  = ma.pi_comp_cons_hist;  end
        fp = fopen(progf,'a'); fprintf(fp,'cell %d done (%.1f s/步均)\n', k, ...
            sum(t_samp(:,1,k),'omitnan')/50); fclose(fp);
    catch err
        arm_failed(k) = true; arm_errmsg{1,k} = sprintf('%s | %s', err.identifier, err.message);
        fprintf('[%s/c%d] ★失败: %s\n', ARM_NAME, k, arm_errmsg{1,k});
        fp = fopen(progf,'a'); fprintf(fp,'cell %d FAILED: %s\n', k, err.message); fclose(fp);
    end
    save(ckpt, 'rmse_resp','rmae_resp','acc_arr','t_samp','alpha_hist','cons_hist', ...
         'picks','channel_log','arm_failed','arm_errmsg','seeds','ARM_NAME','-v7');
end
fprintf('=== %s 全部 %d cells 完成 ===\n', ARM_NAME, nS);
fp = fopen(progf,'a'); fprintf(fp,'ALLDONE\n'); fclose(fp);
exit(0);

function s = ternary(c, a, b)
if c, s = a; else, s = b; end
end
