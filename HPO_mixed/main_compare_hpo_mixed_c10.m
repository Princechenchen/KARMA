clear; close all;
[parent_dir,~,~] = fileparts(pwd);
addpath([parent_dir,'/src']); addpath([parent_dir,'/sampling_Funciton']);
addpath([parent_dir,'/numeric_sample']);
addpath([parent_dir,'/testFunc']); addpath([parent_dir,'\grid']);
addpath([parent_dir,'\MRSM_subprograms1']); addpath([parent_dir,'\MS-PSO-pro']); addpath(parent_dir);
PYEXE = 'F:\tools\Python39\python.exe';
pe = pyenv('Version', PYEXE);
fprintf('pyenv: %s (Version %s)\n', pe.Executable, pe.Version);
insert(py.sys.path, int32(0), pwd);
TAU = 0.05;
c10_mc_samples = 5000; c10_seed = 2026; c10_loo_exact = false;
hhs_failure_class = 1;
pi_unif_grid = 4;
resume_if_checkpoint = true;
cfg = struct('c10_mc_samples',c10_mc_samples, 'c10_seed',c10_seed, ...
             'c10_loo_exact',c10_loo_exact, 'hhs_failure_class',hhs_failure_class, ...
             'pi_unif_grid',pi_unif_grid);
progf = fullfile(pwd,'hpo_mixed_c10_prog.txt');
fp = fopen(progf,'w'); fprintf(fp,'hpo mixed start %s\n', datetime('now')); fclose(fp);
BE = load(fullfile(pwd,'rec_hpo_base.mat'));
CHAIN_CELLS = [10];  base10 = BE.base10(CHAIN_CELLS);
nS = numel(base10);
seeds = arrayfun(@(b) b.rs, base10);
assert(abs(BE.TAU - TAU) < 1e-12, '基座 TAU 与驱动 TAU 不一致——τ 校准后须同改两处');
BK = struct('X_bank',double(BE.X_bank), 'y1_bank',double(BE.y1_bank), ...
            'gap_bank',double(BE.gap_bank));
assert(isequal(size(BK.X_bank),[12000,8]), 'bank 形状异常');
fp = fopen(progf,'a'); fprintf(fp,'cells 1-%d seeds=[%s] TAU=%.3f\n', ...
    nS, num2str(seeds), TAU); fclose(fp);
arm_names = {'C1-MaximinLHS','C2-MeanMSE','C3-IMSE','C4-PAL-ROI-SUR','C5-AKMCS-IRS', ...
             'C6-piBO-global','C7-piBO-ROI','C9-KeyRegion','C10-Voronoi','C11-HHs', ...
             'C12-QBC-global','C13-BO-EI','C8uc-box3-COMP'};
arm_samplers = {@mLHS_maximin_pool, @mMeanMSE_norm, @mIMSE_refset, @mSUR_ROI2wvarc_unified, ...
                @mAK_MCS_IRS_unified, @mGlobal_SUR2w_varcs02lPD_fast_piBO, ...
                @mROI_SUR2w_varcs02lPD_fast_piBO_DS, @KeyRegion_sampling, ...
                @Voronoi_MixedMetric, @HHs_Mixed, @mGlobal_SUR2w_varcs02lPD_fast_QBC, ...
                @mGlobal_SUR2w_varcs02lPD_fast_EI, @mROI_SUR2w_varcs02lPD_fast_PI_pil3_sob_DS};
arm_attach = {'none','none','none','none','none','pibo','pibo','none','voronoi','hhs', ...
              'qbc','none','comp'};
POOL_ARM = logical(ones(1, 13));
nA = numel(arm_names);
assert(numel(arm_samplers)==nA && numel(arm_attach)==nA && numel(POOL_ARM)==nA, '臂表长度不一致');
base = struct();
base.variables.ns = 200; base.variables.dim = 8; base.dim = 8;
base.ClassPos = [1,1,2];
base.total_iter = 70; base.n_init = 20;
ccols = find(base.ClassPos==1); nRC = numel(ccols);
resp_lab = {'y1 = val_acc','y2 = gen_gap'};
assert(numel(resp_lab)==nRC, 'resp_lab 与连续响应数不一致');
[pi_hg_fn, pi_bnd_fn, pi_merged, PC] = pi_hpo_prior();
arm_priors = cell(1, nA);
arm_priors{6} = pi_merged;
arm_priors{7} = pi_merged;
arm_comp = cell(1, nA);
arm_comp{13} = {pi_hg_fn, pi_bnd_fn};
fprintf('===== [hpo-mixed] 先验几何 v3: 甜区[%.2f,%.2f] 过拟合盒 x1>=%.2f,x2>=%.2f 壳厚=%.2f =====\n', ...
    PC.good_lo, PC.good_hi, PC.ov_lo(1), PC.ov_lo(2), PC.shell_tg);
fp = fopen(progf,'a'); fprintf(fp,'prior geometry done\n'); fclose(fp);
n_seq = base.total_iter - base.n_init;
rmse_resp = nan(n_seq, nA, nS, nRC);  rmae_resp = nan(n_seq, nA, nS, nRC);
acc_arr = nan(n_seq, nA, nS);
t_samp    = nan(n_seq, nA, nS);
bank_hits = zeros(nA, nS);  online_hits = zeros(nA, nS);
alpha_hist = cell(nA, nS);  cons_hist = cell(nA, nS);
picks      = cell(nA, nS);
arm_failed = false(nA, nS);  arm_errmsg = cell(nA, nS);
ckpt = fullfile(pwd,'unified_hpo_mixed_ckpt_c10.mat');
start_si = 1;
if resume_if_checkpoint && isfile(ckpt)
    C = load(ckpt);
    needf = {'rmse_resp','rmae_resp','acc_arr','t_samp','alpha_hist','cons_hist', ...
             'picks','arm_failed','arm_errmsg','seeds','bank_hits','online_hits'};
    if all(isfield(C,needf)) && numel(C.seeds)<=nS && isequal(C.seeds,seeds(1:numel(C.seeds))) ...
            && isequal(size(C.rmse_resp,2),nA) && isequal(size(C.rmse_resp,4),nRC)
        k = numel(C.seeds);
        d_all = reshape(C.acc_arr(end,:,:), nA, k);  f_all = reshape(logical(C.arm_failed), nA, k);
        doneSi = find(all(~isnan(d_all) | f_all, 1));
        if ~isempty(doneSi)
            rmse_resp(:,:,1:k,:) = C.rmse_resp;  rmae_resp(:,:,1:k,:) = C.rmae_resp;
            acc_arr(:,:,1:k) = C.acc_arr;  t_samp(:,:,1:k) = C.t_samp;
            alpha_hist(:,1:k) = C.alpha_hist;  cons_hist(:,1:k) = C.cons_hist;
            picks(:,1:k) = C.picks;
            bank_hits(:,1:k) = C.bank_hits;  online_hits(:,1:k) = C.online_hits;
            arm_failed(:,1:k) = C.arm_failed; arm_errmsg(:,1:k) = C.arm_errmsg;
            start_si = max(doneSi) + 1;
            if start_si <= nS
                fprintf('续跑: 重复 1-%d 已完成,从重复 %d (cell %d, rs=%d) 开始\n', ...
                    max(doneSi), start_si, start_si, seeds(start_si));
            else
                fprintf('续跑: 全部重复已完成,直接汇总\n');
            end
        end
    end
end
for si = start_si:nS
    b = base10(si);
    model = base;
    model.rs         = b.rs;
    model.candidate  = b.candidate0;
    model.candidate0 = b.candidate0;
    model.testsample = b.testsample;
    model.testset    = b.testset;
    model.init_x     = b.init_x;
    model.init_value = b.init_value;
    ytest_fixed      = [BK.y1_bank(b.testset_idx), BK.gap_bank(b.testset_idx)];
    ytest_fixed      = [ytest_fixed, double(ytest_fixed(:,2) > TAU)];
    model.MMRGP      = MMRGP0_correxp(model.init_value, model.init_x, model.ClassPos, model.dim);
    model.now_value  = model.init_value; model.now_x = model.init_x;
    rng(b.rs*100 + CHAIN_CELLS(si));  rng_state = rng;
    fp = fopen(progf,'a'); fprintf(fp,'rep %d (cell %d, rs=%d) base ready (init 20 rows)\n', ...
        si, si, b.rs); fclose(fp);
    for ai = 1:nA
        if arm_failed(ai, si), continue; end
        rng(rng_state);
        ma = model;
        ma = attach_arm(ma, arm_attach{ai}, arm_priors{ai}, arm_comp{ai}, cfg);
        fp = fopen(progf,'a'); fprintf(fp,'arm %s start (r%d)\n', arm_names{ai}, si); fclose(fp);
        t_arm = tic;
        bh0 = 0; oh0 = 0;
        try
            for i = 1:n_seq
                tt = tic;
                [x_star, ma] = arm_samplers{ai}(ma);
                t_samp(i, ai, si) = toc(tt);
                [f_star, src] = hpo_truth(x_star, BK, TAU, CHAIN_CELLS(si));
                if strcmp(src,'bank'), bh0 = bh0 + 1; else, oh0 = oh0 + 1; end
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
                if mod(i,10)==0 || i==1
                    fprintf('[%s/r%d] iter %d rmae_y=[%.4f %.4f] acc=%.4f (%s)\n', ...
                        arm_names{ai}, si, i, rmae_resp(i,ai,si,1), rmae_resp(i,ai,si,2), ...
                        acc_arr(i,ai,si), src);
                end
                fp = fopen(progf,'a'); fprintf(fp, 'r%d %s step %d ok (%.1f s, %s)\n', ...
                    si, arm_names{ai}, i, t_samp(i,ai,si), src); fclose(fp);
            end
            bank_hits(ai,si) = bh0; online_hits(ai,si) = oh0;
            if POOL_ARM(ai)
                assert(oh0 == 0, '池臂 %s 触发在线评估(r%d)——协议违约, 需排查', arm_names{ai}, si);
            elseif bh0 > 0
                fprintf('[%s/r%d] ⚠自主臂 %d 步命中 bank 行(概率极低, 记录备查)\n', ...
                    arm_names{ai}, si, bh0);
            end
            if isfield(ma,'pi_comp_alpha_hist'), alpha_hist{ai,si} = ma.pi_comp_alpha_hist; end
            if isfield(ma,'pi_comp_cons_hist'),  cons_hist{ai,si}  = ma.pi_comp_cons_hist;  end
            fprintf('[%s/r%d] 整臂耗时 %.1f min (采样器均值 %.1f s/步; bank %d / online %d)\n', ...
                arm_names{ai}, si, toc(t_arm)/60, ...
                sum(t_samp(:,ai,si),'omitnan')/sum(~isnan(t_samp(:,ai,si))), bh0, oh0);
            fp = fopen(progf,'a'); fprintf(fp,'arm %s done (r%d) bank=%d online=%d\n', ...
                arm_names{ai}, si, bh0, oh0); fclose(fp);
        catch err
            arm_failed(ai, si) = true;
            arm_errmsg{ai, si} = sprintf('%s | %s', err.identifier, err.message);
            fprintf('[%s/r%d] ★失败: %s\n', arm_names{ai}, si, arm_errmsg{ai,si});
            fp = fopen(progf,'a'); fprintf(fp,'arm %s FAILED (r%d): %s\n', ...
                arm_names{ai}, si, err.message); fclose(fp);
        end
        save(ckpt, 'rmse_resp','rmae_resp','acc_arr','t_samp','alpha_hist','cons_hist', ...
             'picks','arm_failed','arm_errmsg','arm_names','seeds','n_seq','resp_lab', ...
             'bank_hits','online_hits','TAU','-v7');
    end
    fprintf('=== rep %d/%d (rs=%d) done ===\n', si, nS, b.rs);
    fp = fopen(progf,'a'); fprintf(fp,'rep %d done\n', si); fclose(fp);
end
ok = ~arm_failed & ~isnan(reshape(acc_arr(end,:,:), nA, nS));
for j = 1:nRC
    fprintf('\n===== 连续响应 %s 末步 RMAE 均值±std + AULC, seeds=%s =====\n', ...
        resp_lab{j}, mat2str(seeds));
    for ai = 1:nA
        if ~any(ok(ai,:))
            em = arm_errmsg(ai,:); em = em(~cellfun(@isempty, em));
            if isempty(em), em = {'无数据(该臂未在本轮运行)'}; end
            fprintf('%-22s ★失败/无数据: %s\n', arm_names{ai}, strjoin(em,' ; '));
            continue;
        end
        r = squeeze(rmae_resp(:,ai,:,j));  r = r(:,ok(ai,:));
        fprintf('%-22s %6.4f±%.4f  AULC %8.4f\n', arm_names{ai}, ...
            mean(r(end,:)), std(r(end,:)), mean(mean(r,1)));
    end
end
fprintf('\n===== 分类响应(过拟合标签) ACC 末步 均值±std =====\n');
for ai = 1:nA
    if ~any(ok(ai,:)), continue; end
    a = squeeze(acc_arr(:,ai,:));  a = a(:,ok(ai,:));
    fprintf('%-22s %6.4f±%.4f\n', arm_names{ai}, mean(a(end,:)), std(a(end,:)));
end
fprintf('\n真值路径审计: bank_hits 总 %d / online_hits 总 %d (池臂在线应为 0)\n', ...
    sum(bank_hits(:)), sum(online_hits(:)));
fprintf('主循环完成: checkpoint=%s\n', ckpt);
fp = fopen(progf,'a'); fprintf(fp,'ALLDONE\n'); fclose(fp);
exit(0);
function m = attach_arm(m, kind, prior, comp, cfg)
switch kind
    case 'none'
    case 'pibo'
        if isempty(prior)
            error('attach_arm:MissingPrior','pibo 臂需要非空先验句柄');
        end
        m.pi_phys_fn = prior;
        m.pibo_beta  = 3;
    case 'comp'
        if numel(comp)~=2
            error('attach_arm:BadComp','comp 需为 {π_hg,π_bnd} 二元 cell');
        end
        m.pi_comp_fns      = comp;
        m.pi_comp_adaptive = true;
        m.pi_comp_eta      = 0.5;
        m.pi_comp_alpha_min = 0.1;
        m.pi_comp_alpha    = [1 1];
        m.pi_comp_alpha_hist = zeros(0,2);
        m.pi_comp_cons_hist  = zeros(0,2);
        m.pi_unif_grid     = cfg.pi_unif_grid;
        m.pi_unif_sobol_n  = 8192;
    case 'voronoi'
        m.c10_mc_samples = cfg.c10_mc_samples;
        m.c10_seed       = cfg.c10_seed;
        m.c10_loo_exact  = cfg.c10_loo_exact;
    case 'hhs'
        m.hhs_failure_class = cfg.hhs_failure_class;
    case 'qbc'
        m.qbc_K   = 8;
        m.qbc_frac = 0.8;
    otherwise
        error('attach_arm:UnknownKind','未知 attach 类型: %s', kind);
end
end
function [f3, src] = hpo_truth(x_row, BK, TAU, cell_id)
[tf, row] = ismember(x_row, BK.X_bank, 'rows');
if tf
    y1 = BK.y1_bank(row);  gp = BK.gap_bank(row);  src = 'bank';
else
    r = hpo_eval_online(x_row, cell_id);
    y1 = r.y1;  gp = r.gap;  src = 'online';
end
f3 = [y1, gp, double(gp > TAU)];
end
