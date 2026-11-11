clear; close all;
nw = 4;
si_list = 1:6;
[parent_dir,~,~] = fileparts(pwd);
assert(endsWith(pwd,'wingweight'), '请在 wingweight/ 目录下运行本脚本');
this_dir = pwd;
addpath([parent_dir,'/src']); addpath([parent_dir,'/sampling_Funciton']);
addpath([parent_dir,'/testFunc']); addpath([parent_dir,'\grid']);
addpath([parent_dir,'\MRSM_subprograms1']); addpath([parent_dir,'\MS-PSO-pro']);
addpath([parent_dir,'\tenbar']); addpath(parent_dir);
assert(contains(which('MMRGP0'),'wingweight'), 'MMRGP0 未解析到 wingweight/');
w_lp = which('mROI_SUR2w_varcs02lPD');
assert(contains(w_lp,'sampling_Funciton'), 'lPD 采样器未解析到 sampling_Funciton(4月版): %s', w_lp);
lpd_sampler = @mROI_SUR2w_varcs02lPD;
assert(contains(functions(lpd_sampler).file,'sampling_Funciton'), 'lPD 句柄绑定失败');
addpath([parent_dir,'\numeric_sample']);
w_lp2 = which('mROI_SUR2w_varcs02lPD');
fprintf('lPD 绑定核验: 句柄=%s | 当前 which=%s (numeric_sample 遮蔽生效, 但句柄已绑 4 月版)\n', ...
    functions(lpd_sampler).file, w_lp2);
need = {'mLHS_maximin','meanMSE','IMSEs','mSUR_ROI2wvarc','AK_MCS_IRS', ...
        'KeyRegion_sampling','Voronoi_MixedMetric','HHs_Mixed', ...
        'pi_wingweight_prior','mROI_SUR2w_varcs02lPD_fast_PI_comp_DS_b3u', ...
        'loo_consistency_uniform_comp_uconf_sobol','mopso_adaptive_elitism_crowding_PI3', ...
        'prior_anneal_factor','sobolset','cand_gen','wingweight'};
for k = 1:numel(need)
    assert(~isempty(which(need{k})), '组件缺失: %s (路径未挂载)', need{k});
end
pyenv('Version', 'F:\tools\Python39\python.exe');
pe0 = pyenv;
fprintf('client python: %s | %s\n', pe0.Version, pe0.Executable);
[pi_hg_fn, pi_bnd_fn, pri_snap] = pi_wingweight_prior();
fprintf('C8 先验 v1: 3 软锚(tc<0.30 | Nz&Wdg<0.35 | A<0.25, ew=%.2f) + 双裕度脊 σ=(%.3f, %.3f)\n', ...
    pri_snap.ew, pri_snap.sigma1, pri_snap.sigma2);
B = load(fullfile(pwd,'ext6_ww_base.mat'), 'ext6base', 'seeds');
ext6base = B.ext6base;  seeds = B.seeds(si_list);  clear B
nS = numel(si_list);
fprintf('ext6 基座: seeds=%s (si=%s)\n', mat2str(seeds), mat2str(si_list));
arm_names = {'C1-MaximinLHS','C2-MeanMSE','C3-IMSE','C4-SUR-ROI','C5-AKMCS-IRS', ...
             'lPD','C9-KeyRegion','C10-Voronoi','C11-HHs','C8uc-box3u-COMP'};
arm_samplers = {@mLHS_maximin, @meanMSE, @IMSEs, @mSUR_ROI2wvarc, ...
                @(mm) AK_MCS_IRS(mm, @(x)wingweight(x)), ...
                lpd_sampler, ...
                @KeyRegion_sampling, @Voronoi_MixedMetric, @HHs_Mixed, ...
                @mROI_SUR2w_varcs02lPD_fast_PI_comp_DS_b3u};
arm_attach  = {'none','none','none','none','none','none','none','voronoi','hhs','c8'};
nA = numel(arm_names);
assert(numel(arm_samplers)==nA && numel(arm_attach)==nA, '臂表长度不一致');
obj_fct = @(x)wingweight(x);
base = struct();
base.variables.ns = 200; base.variables.dim = 10; base.dim = 10;
base.ClassPos = [1,1,2];
base.total_iter = 45; base.n_init = 25;
ccols = find(base.ClassPos==1);
nRC   = numel(ccols);
resp_lab = {'y1=机翼重量(y1<=250)','y2=翼段指标(y2<=50)'};
assert(numel(resp_lab)==nRC, 'resp_lab 与连续响应数不一致');
c10_mc_samples = 5000;  c10_seed = 2026;  c10_loo_exact = false;  hhs_failure_class = 0;
pi_unif_sobol_n = 4000;
pi3_anneal_cfg  = struct('k', 5, 'L', 3, 'tau_drop', 0.25);
n_seq = base.total_iter - base.n_init;
rmse_resp = nan(n_seq, nA, nS, nRC);  rmae_resp = nan(n_seq, nA, nS, nRC);
acc_arr = nan(n_seq, nA, nS);
t_samp    = nan(n_seq, nA, nS);
picks      = cell(nA, nS);
arm_failed = false(nA, nS);  arm_errmsg = cell(nA, nS);
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('Processes', nw, 'IdleTimeout', 120);
else
    fprintf('复用现有 parpool(%d worker)\n', pool.NumWorkers);
end
fprintf('===== wingweight ext6 全方法轮启动: %d worker x %d seeds x %d 臂 x %d 步 =====\n', ...
    pool.NumWorkers, nS, nA, n_seq);
parfor sidx = 1:nS
    cd(this_dir);
    si = si_list(sidx);
    rs = double(seeds(sidx));
    acc_s    = nan(n_seq, nA);      tsamp_s  = nan(n_seq, nA);
    rmae_s   = nan(n_seq, nA, nRC); rmse_s   = nan(n_seq, nA, nRC);
    picks_s  = cell(nA, 1);
    failed_s = false(nA, 1);        errmsg_s = cell(nA, 1);
    hj = ext6base{sidx};
    assert(size(hj.candidate0,1)==200 && size(hj.testset,1)==5000 ...
        && size(hj.init_x,1)==25 && size(hj.testsample,1)==12000, 'ext6 基座尺寸异常');
    model = base;
    model.rs         = rs;
    model.candidate  = hj.candidate0;
    model.candidate0 = hj.candidate0;
    model.testsample = hj.testsample;
    model.testset    = hj.testset;
    model.init_x     = hj.init_x;
    model.init_value = hj.init_value;
    ytraw   = wingweight(model.testset);
    my_cell = mean(ytraw);  my_cell(model.ClassPos==2) = 1;
    ytest_fixed = ytraw ./ my_cell;
    iverr = max(abs(wingweight(hj.init_x)./my_cell - hj.init_value), [], 'all');
    fprintf('[seed s%d-rs=%d] my=[%s] | init 指纹 max|D|=%.2e | 正类=%.4f\n', ...
        si, rs, mat2str(my_cell,4), iverr, mean(ytraw(:, find(base.ClassPos==2))));
    assert(iverr < 1e-9, 'init_value 归一化指纹不匹配(seed %d)', si);
    model.MMRGP      = MMRGP0(model.init_value, model.init_x, model.ClassPos, model.dim);
    model.now_value  = model.init_value; model.now_x = model.init_x;
    rng_state = rng(rs*100 + si);
    hj = [];
    for ai = 1:nA
        if failed_s(ai), continue; end
        rng(rng_state);
        ma = model;
        switch arm_attach{ai}
            case 'voronoi'
                ma.c10_mc_samples = c10_mc_samples;
                ma.c10_seed       = c10_seed;
                ma.c10_loo_exact  = c10_loo_exact;
            case 'hhs'
                ma.hhs_failure_class = hhs_failure_class;
            case 'c8'
                ma.pi_comp_fns      = {pi_hg_fn, pi_bnd_fn};
                ma.pi_comp_adaptive = true;
                ma.pi_comp_eta      = 0.5;
                ma.pi_comp_alpha_min = 0.1;
                ma.pi_comp_alpha    = [1 1];
                ma.pi_comp_alpha_hist = zeros(0,2);
                ma.pi_comp_cons_hist  = zeros(0,2);
                ma.pi_unif_sobol_n   = pi_unif_sobol_n;
                ma.pi3_anneal        = pi3_anneal_cfg;
        end
        t_arm = tic;
        try
            for i = 1:n_seq
                tt = tic;
                [x_star, ma] = arm_samplers{ai}(ma);
                tsamp_s(i, ai) = toc(tt);
                if ai == nA
                    [~, xind] = ismember(x_star, model.candidate0, 'rows');
                    assert(xind > 0, 'C8 x_star 不在候选池内(ismember 失败)');
                end
                f_star = wingweight(x_star);
                f_star = f_star ./ my_cell;
                ma.now_value = [ma.now_value; f_star]; ma.now_x = [ma.now_x; x_star];
                ma.MMRGP = MMRGP0(ma.now_value, ma.now_x, ma.ClassPos, ma.dim);
                [ypred] = predict_resp(ma.MMRGP, model.testset);
                ev = error_func(ytest_fixed, ypred, ma.ClassPos);
                for j = 1:nRC
                    rmae_s(i, ai, j) = ev{1}(j);
                    rmse_s(i, ai, j) = sqrt(mean((ytest_fixed(:,ccols(j)) - ypred(:,ccols(j))).^2));
                end
                acc_s(i, ai)     = mean(ev{4});
                picks_s{ai}(i,:) = x_star;
                if mod(i,10)==0 || i==1
                    fprintf('[%s/s%d-rs=%d] iter %d rmse_y=[%.4f %.4f] acc=%.4f\n', ...
                        arm_names{ai}, si, rs, i, ...
                        rmse_s(i,ai,1), rmse_s(i,ai,2), acc_s(i,ai));
                end
            end
            fprintf('[%s/s%d-rs=%d] 整臂 %.1f min\n', arm_names{ai}, si, rs, toc(t_arm)/60);
        catch err
            failed_s(ai) = true;
            errmsg_s{ai} = sprintf('%s | %s', err.identifier, err.message);
            fprintf('[%s/s%d-rs=%d] ★失败: %s\n', arm_names{ai}, si, rs, errmsg_s{ai});
        end
    end
    rmse_resp(:,:,sidx,:) = rmse_s;   rmae_resp(:,:,sidx,:) = rmae_s;
    acc_arr(:,:,sidx)     = acc_s;    t_samp(:,:,sidx)      = tsamp_s;
    picks(:,sidx)         = picks_s;
    arm_failed(:,sidx)    = failed_s; arm_errmsg(:,sidx)    = errmsg_s;
    fprintf('=== [s%d-rs=%d] seed done ===\n', si, rs);
end
ckpt = fullfile(this_dir, sprintf('unified_wingweight_ext6_s%d_%d_checkpoint.mat', ...
    si_list(1), si_list(end)));
save(ckpt, 'rmse_resp','rmae_resp','acc_arr','t_samp','picks','arm_failed','arm_errmsg', ...
     'arm_names','seeds','si_list','n_seq','resp_lab','pi_unif_sobol_n','pi3_anneal_cfg','-v7');
fprintf('parfor 完成: 合并 checkpoint=%s\n', ckpt);
ok = ~arm_failed & ~isnan(reshape(acc_arr(end,:,:), nA, nS));
for j = 1:nRC
    fprintf('\n===== 连续响应 %s 末步(第 %d 步) RMAE 均值±std + AULC =====\n', resp_lab{j}, n_seq);
    fprintf('%-22s %14s %12s %10s\n','arm','RMAE','AULC_rmae','s/step');
    for ai = 1:nA
        if ~any(ok(ai,:))
            fprintf('%-22s ★失败/无数据: %s\n', arm_names{ai}, strjoin(arm_errmsg(ai,:),' ; '));
            continue;
        end
        r = rmae_resp(:,ai,:,j);  r = r(:,ok(ai,:));
        t = t_samp(:,ai,:);       t = t(:,ok(ai,:));
        fprintf('%-22s %6.4f±%.4f  %8.4f  %8.2f\n', arm_names{ai}, ...
            mean(r(end,:)), std(r(end,:)), mean(mean(r,1)), mean(t(:)));
    end
end
fprintf('\n===== 分类响应(yc=y1<=250 且 y2<=50) ACC 末步 均值±std =====\n');
fprintf('%-22s %14s\n','arm','ACC');
for ai = 1:nA
    if ~any(ok(ai,:)), continue; end
    a = acc_arr(:,ai,:);  a = a(:,ok(ai,:));
    fprintf('%-22s %6.4f±%.4f\n', arm_names{ai}, mean(a(end,:)), std(a(end,:)));
end
fprintf('\next6 轮主循环完成: %s\n', ckpt);
fprintf('=== main_compare_wingweight_ext6_par DONE OK ===\n');
