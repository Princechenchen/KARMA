clear; close all; clc;
SMOKE = ~isempty(getenv('WW_PIBO_S10_SMOKE'));
FORCE_RERUN = false;
TOKEN = 'main_compare_wingweight_pibo_sub10';
[parent_dir,~,~] = fileparts(pwd);
assert(endsWith(pwd,'wingweight'), '请在 wingweight/ 目录下运行本脚本');
this_dir = pwd;
addpath([parent_dir,'/src']);
addpath([parent_dir,'/sampling_Funciton']);
addpath([parent_dir,'/testFunc']);
addpath([parent_dir,'\grid']);
addpath([parent_dir,'\MRSM_subprograms1']);
addpath([parent_dir,'\MS-PSO-pro']);
addpath(parent_dir); addpath(this_dir);
addpath([parent_dir,'\numeric_sample']);
fp = {'MMRGP0','MRSM','mGlobal_SUR2w_varcs02lPD_fast_piBO', ...
      'mROI_SUR2w_varcs02lPD_fast_piBO','pibo_prior_weight', ...
      'pi_wingweight_prior','gen_ROIs','mopso_adaptive_elitism_crowding', ...
      'prepare_roi_predict_cache','poschol','predict_resp','error_func','wingweight'};
fprintf('===== 路径指纹 =====\n');
for k = 1:numel(fp)
    fprintf('%-42s -> %s\n', fp{k}, which(fp{k}));
    assert(~isempty(which(fp{k})), '组件缺失: %s (路径未挂载)', fp{k});
end
assert(endsWith(which('MRSM'), ['MRSM_subprograms1' filesep 'MRSM.m']), ...
    'MRSM 未解析到 MRSM_subprograms1(corrgauss 基线)');
assert(endsWith(which('mGlobal_SUR2w_varcs02lPD_fast_piBO'), ...
    ['numeric_sample' filesep 'mGlobal_SUR2w_varcs02lPD_fast_piBO.m']), 'C6 采样器路径错');
assert(endsWith(which('mROI_SUR2w_varcs02lPD_fast_piBO'), ...
    ['numeric_sample' filesep 'mROI_SUR2w_varcs02lPD_fast_piBO.m']), 'C7 采样器路径错(应非 DS 版)');
assert(endsWith(which('pi_wingweight_prior'), ['src' filesep 'pi_wingweight_prior.m']), '先验函数路径错');
assert(endsWith(which('mopso_adaptive_elitism_crowding'), ...
    ['MS-PSO-pro' filesep 'mopso_adaptive_elitism_crowding.m']), 'plain pro mopso 路径错');
if SMOKE, diary('smoke_wingweight_pibo_sub10_log.txt'); else, diary('wingweight_pibo_sub10_log.txt'); end
diary on
fprintf('===== %s 启动 %s | MATLAB %s =====\n', TOKEN, datestr(now,'yyyy-mm-dd HH:MM:SS'), version);
prog_file = fullfile(this_dir,'wingweight_pibo_sub10_prog.txt');
if SMOKE, prog_file = fullfile(this_dir,'smoke_wingweight_pibo_sub10_prog.txt'); end
prog_fid = fopen(prog_file, 'w');
fprintf(prog_fid, 'START %s %s\n', TOKEN, datestr(now,'yyyy-mm-dd HH:MM:SS'));
fclose(prog_fid);
rsl      = [29 47 66 84 102 74 110 188 205 226];
base_of  = {'ext6','ext6','ext6','ext6','ext6','ext4','ext4b','ext4c','ext4c','ext4d'};
base_si  = [ 2    3    4    5    6    4    1     3     4    2];
nS = numel(rsl);
assert(numel(base_of)==nS && numel(base_si)==nS, '种子-基座映射长度不一致');
assert(numel(unique(rsl))==nS, 'duplicate subset seeds');
base_names = unique(base_of);
bases = struct();
for b = 1:numel(base_names)
    bf = fullfile(this_dir, sprintf('%s_ww_base.mat', base_names{b}));
    if strcmp(base_names{b}, 'ext6')
        B = load(bf, 'ext6base', 'seeds');
        B = struct('cells', {B.ext6base}, 'seeds', {B.seeds});
    else
        B = load(bf, 'ext4base', 'seeds');
        B = struct('cells', {B.ext4base}, 'seeds', {B.seeds});
    end
    bases.(base_names{b}) = B;
    fprintf('基座 %s: seeds=%s\n', base_names{b}, mat2str(double(B.seeds)));
end
for si = 1:nS
    B = bases.(base_of{si});
    assert(double(B.seeds(base_si(si))) == rsl(si), ...
        'seed %d 在基座 %s#%d 位置不符(档案=%g)', rsl(si), base_of{si}, base_si(si), double(B.seeds(base_si(si))));
end
fprintf('种子基座映射核对通过: rsl=%s\n', mat2str(rsl));
if SMOKE
    nS_run = 1; seq_cap = 3;
    fprintf('===== SMOKE 模式: 1 seed x 3 步, 不落盘 =====\n');
else
    nS_run = nS; seq_cap = inf;
end
[pi_hg_fn, pi_bnd_fn, pri_snap] = pi_wingweight_prior();
pi_merged = @(X) max(pi_hg_fn(X), pi_bnd_fn(X));
fprintf('先验 v1: 3 软锚 + 双裕度脊(ew=%.2f, sigma=%.3f/%.3f) | merged=max, 无仲裁\n', ...
    pri_snap.ew, pri_snap.sigma1, pri_snap.sigma2);
arm_names    = {'C6-piBO-global','C7-piBO-ROI'};
arm_samplers = {@mGlobal_SUR2w_varcs02lPD_fast_piBO, @mROI_SUR2w_varcs02lPD_fast_piBO};
nA = numel(arm_names);
ClassPos = [1,1,2];   dim = 10;
nRC = 2;  ccols = 1:2;
resp_lab = {'y1=机翼重量(y1<=250)','y2=翼段指标(y2<=50)'};
total_iter = 45;  n_init = 25;  n_seq = total_iter - n_init;
pibo_beta = (total_iter - n_init) / 10;
fprintf('pibo_beta = (45-25)/10 = %g\n', pibo_beta);
rmse_resp = nan(n_seq, nA, nS, nRC);  rmae_resp = nan(n_seq, nA, nS, nRC);
acc_arr   = nan(n_seq, nA, nS);       t_samp   = nan(n_seq, nA, nS);
picks = cell(nA, nS);
pibo_prior_c = cell(nA, nS);  pibo_weight_c = cell(nA, nS);  pibo_n_c = cell(nA, nS);
arm_failed = false(nA, nS);  arm_errmsg = cell(nA, nS);
arm_min = nan(nA, nS);
t_all = tic;
for si = 1:nS_run
    cd(this_dir);
    seed_file = fullfile(this_dir, sprintf('unified_wingweight_pibo_sub10_seed%02d.mat', si));
    rs = double(rsl(si));
    if ~SMOKE && ~FORCE_RERUN && exist(seed_file, 'file')
        sv = load(seed_file);
        if sv.nsteps >= n_seq && ~any(sv.failed_a)
            rmse_resp(:,:,si,:) = sv.rmse_s; rmae_resp(:,:,si,:) = sv.rmae_s;
            acc_arr(:,:,si) = sv.acc_s; t_samp(:,:,si) = sv.tsamp_s;
            picks(:,si) = sv.picks_s;
            pibo_prior_c(:,si) = sv.pibo_prior_s; pibo_weight_c(:,si) = sv.pibo_weight_s;
            pibo_n_c(:,si) = sv.pibo_n_s;
            arm_failed(:,si) = sv.failed_a; arm_errmsg(:,si) = sv.errmsg_a;
            arm_min(:,si) = sv.arm_min_c;
            fprintf('=== [s%d/rs=%d] 已有完成断点, 跳过 ===\n', si, rs);
            continue;
        end
        fprintf('[s%d/rs=%d] 断点不完整, 整 seed 重跑\n', si, rs);
    end
    rmse_s = nan(n_seq, nA, nRC); rmae_s = nan(n_seq, nA, nRC);
    acc_s = nan(n_seq, nA); tsamp_s = nan(n_seq, nA);
    picks_s = cell(nA,1); failed_a = false(nA,1); errmsg_a = cell(nA,1);
    arm_min_c = nan(nA,1);
    pibo_prior_s = cell(nA,1); pibo_weight_s = cell(nA,1); pibo_n_s = cell(nA,1);
    hj = bases.(base_of{si}).cells{base_si(si)};
    assert(size(hj.candidate0,1)==200 && size(hj.testset,1)==5000 ...
        && size(hj.init_x,1)==25 && size(hj.testsample,1)==12000, '基座尺寸异常(seed %d)', rs);
    model = struct();
    model.rs = rs;
    model.variables = struct('ns', 200, 'dim', dim);
    model.dim = dim;  model.ClassPos = ClassPos;
    model.total_iter = total_iter;  model.n_init = n_init;
    model.candidate  = hj.candidate0;   model.candidate0 = hj.candidate0;
    model.testsample = hj.testsample;   model.testset    = hj.testset;
    model.init_x     = hj.init_x;       model.init_value = hj.init_value;
    ytraw = wingweight(model.testset);
    my_cell = mean(ytraw);  my_cell(ClassPos==2) = 1;
    ytest_fixed = ytraw ./ my_cell;
    iverr = max(abs(wingweight(hj.init_x)./my_cell - hj.init_value), [], 'all');
    fprintf('[s%d/rs=%d] my=[%s] | init 指纹 max|D|=%.2e | 正类率=%.4f\n', ...
        si, rs, mat2str(my_cell,4), iverr, mean(ytraw(:, find(ClassPos==2))));
    assert(iverr < 1e-9, 'init_value 归一化指纹不匹配(seed %d)', rs);
    model.MMRGP = MMRGP0(model.init_value, model.init_x, ClassPos, dim);
    if isfield(model.MMRGP,'corrfunc')
        assert(model.MMRGP.corrfunc == "corrgauss", '★MMRGP 核不是 corrgauss —— wingweight 口径被破坏');
    else
        fprintf('[s%d] ★警告: MMRGP 无 corrfunc 字段, 核身份未断言(依赖路径指纹)\n', si);
    end
    model.now_value = model.init_value;  model.now_x = model.init_x;
    rng_state = rng(rs*100 + base_si(si));
    hj = [];
    for ai = 1:nA
        if failed_a(ai), continue; end
        rng(rng_state);
        ma = model;
        ma.pi_phys_fn = pi_merged;
        ma.pibo_beta  = pibo_beta;
        t_arm = tic;
        try
            nsteps = min(n_seq, seq_cap);
            for i = 1:nsteps
                tt = tic;
                [x_star, ma] = arm_samplers{ai}(ma);
                tsamp_s(i, ai) = toc(tt);
                if ai == 1
                    [~, xind] = ismember(x_star, model.candidate0, 'rows');
                    assert(xind > 0, 'C6 x_star 不在候选池内(ismember 失败)');
                end
                f_star = wingweight(x_star);
                f_star = f_star ./ my_cell;
                ma.now_value = [ma.now_value; f_star];  ma.now_x = [ma.now_x; x_star];
                ma.MMRGP = MMRGP0(ma.now_value, ma.now_x, ClassPos, dim);
                [ypred] = predict_resp(ma.MMRGP, model.testset);
                ev = error_func(ytest_fixed, ypred, ClassPos);
                for j = 1:nRC
                    rmae_s(i, ai, j) = ev{1}(j);
                    rmse_s(i, ai, j) = sqrt(mean((ytest_fixed(:,ccols(j)) - ypred(:,ccols(j))).^2));
                end
                acc_s(i, ai) = mean(ev{4});
                picks_s{ai}(i,:) = x_star;
                fprintf('[%s/s%d-rs=%d] iter %2d/%d t=%7.1fs | rmse=[%.4f %.4f] rmae=[%.4f %.4f] acc=%.4f\n', ...
                    arm_names{ai}, si, rs, i, nsteps, tsamp_s(i,ai), ...
                    rmse_s(i,ai,1), rmse_s(i,ai,2), rmae_s(i,ai,1), rmae_s(i,ai,2), acc_s(i,ai));
                prog_fid = fopen(prog_file, 'a');
                fprintf(prog_fid, 'PROG %s s%d/rs=%d %s iter %d/%d\n', ...
                    datestr(now,'yyyy-mm-dd HH:MM:SS'), si, rs, arm_names{ai}, i, nsteps);
                fclose(prog_fid);
            end
            arm_min_c(ai) = toc(t_arm)/60;
            fprintf('[%s/s%d-rs=%d] 整臂 %.1f min\n', arm_names{ai}, si, rs, arm_min_c(ai));
            pibo_prior_s{ai} = ma.pibo_last_prior;  pibo_weight_s{ai} = ma.pibo_last_weight;
            pibo_n_s{ai} = ma.pibo_last_n;
        catch err
            failed_a(ai) = true;
            errmsg_a{ai} = sprintf('%s | %s', err.identifier, err.message);
            fprintf('[%s/s%d-rs=%d] ★失败: %s\n', arm_names{ai}, si, rs, errmsg_a{ai});
            for st = err.stack'
                fprintf('    at %s:%d\n', st.name, st.line);
            end
        end
    end
    rmse_resp(:,:,si,:) = rmse_s;  rmae_resp(:,:,si,:) = rmae_s;
    acc_arr(:,:,si) = acc_s;       t_samp(:,:,si) = tsamp_s;
    picks(:,si) = picks_s;
    pibo_prior_c(:,si) = pibo_prior_s;  pibo_weight_c(:,si) = pibo_weight_s;
    pibo_n_c(:,si) = pibo_n_s;
    arm_failed(:,si) = failed_a;   arm_errmsg(:,si) = errmsg_a;
    arm_min(:,si) = arm_min_c;
    if ~SMOKE
        nsteps = sum(~isnan(rmse_s(:,1)));
        base_of_seed = base_of{si};   base_pos = base_si(si);
        close all;
        save(seed_file, 'rmse_s','rmae_s','acc_s','tsamp_s','picks_s', ...
             'pibo_prior_s','pibo_weight_s','pibo_n_s', ...
             'failed_a','errmsg_a','arm_min_c','rs','nsteps','base_of_seed','base_pos', '-v7');
        fprintf('=== [s%d/rs=%d] seed done(断点已存, nsteps=%d) | 总耗时 %.1f min ===\n', ...
            si, rs, nsteps, toc(t_all)/60);
        prog_fid = fopen(prog_file, 'a');
        fprintf(prog_fid, 'SEEDDONE %s s%d/rs=%d nsteps=%d\n', ...
            datestr(now,'yyyy-mm-dd HH:MM:SS'), si, rs, nsteps);
        fclose(prog_fid);
    end
end
if ~SMOKE
    close all;
    ckpt = fullfile(this_dir,'unified_wingweight_pibo_sub10_checkpoint.mat');
    save(ckpt, 'rmse_resp','rmae_resp','acc_arr','t_samp','picks', ...
         'pibo_prior_c','pibo_weight_c','pibo_n_c', ...
         'arm_failed','arm_errmsg','arm_names','rsl','n_seq','resp_lab','arm_min', ...
         'base_of','base_si','pibo_beta','-v7');
    fprintf('主循环完成: 合并 checkpoint=%s\n', ckpt);
end
nsteps_eff = min(n_seq, seq_cap);
ok = ~arm_failed(:,1:nS_run) & ~isnan(reshape(acc_arr(nsteps_eff,:,1:nS_run), nA, nS_run));
fprintf('\n===== wingweight C6/C7 piBO·corrgauss·10 seeds 末步(第 %d 步) 均值±std =====\n', nsteps_eff);
fprintf('%-16s %18s %18s %14s %10s\n','arm','RMAE_y1','RMAE_y2','ACC','s/step');
for ai = 1:nA
    if ~any(ok(ai,:))
        msg = arm_errmsg(ai,1:nS_run);  msg = msg(~cellfun(@isempty,msg));
        if isempty(msg), msg = {'无数据'}; end
        fprintf('%-16s ★失败/无数据: %s\n', arm_names{ai}, strjoin(msg,' ; '));
        continue;
    end
    r1 = reshape(rmae_resp(nsteps_eff,ai,1:nS_run,1),1,[]);  r1 = r1(ok(ai,:));
    r2 = reshape(rmae_resp(nsteps_eff,ai,1:nS_run,2),1,[]);  r2 = r2(ok(ai,:));
    ac = reshape(acc_arr(nsteps_eff,ai,1:nS_run),1,[]);      ac = ac(ok(ai,:));
    t  = reshape(t_samp(1:nsteps_eff,ai,1:nS_run),nsteps_eff,[]);  t = t(:,ok(ai,:));
    fprintf('%-16s %6.4f±%.4f %12.4f±%.4f %8.4f±%.4f %8.2f\n', arm_names{ai}, ...
        mean(r1), std(r1), mean(r2), std(r2), mean(ac), std(ac), mean(t(:)));
    a1 = reshape(mean(rmae_resp(:,ai,1:nS_run,1),1),1,[]);   a1 = a1(ok(ai,:));
    a2 = reshape(mean(rmae_resp(:,ai,1:nS_run,2),1),1,[]);   a2 = a2(ok(ai,:));
    fprintf('%-16s AULC y1=%.4f | y2=%.4f\n', '  └', mean(a1), mean(a2));
end
prog_fid = fopen(prog_file, 'a');
fprintf(prog_fid, 'ALLDONE %s %s\n', TOKEN, datestr(now,'yyyy-mm-dd HH:MM:SS'));
fclose(prog_fid);
fprintf('===== %s DONE %s | 总耗时 %.1f min =====\n', TOKEN, ...
    datestr(now,'yyyy-mm-dd HH:MM:SS'), toc(t_all)/60);
diary off
