clear; close all; clc;
SMOKE = ~isempty(getenv('SHIP_PIBO_S10_SMOKE'));
FORCE_RERUN = false;
[parent_dir,~,~] = fileparts(pwd);
assert(endsWith(pwd,'Radar'), '请在 Radar/ 目录下运行本脚本');
this_dir = pwd;
addpath([parent_dir,'/src']); addpath([parent_dir,'/sampling_Funciton']);
addpath([parent_dir,'/testFunc']); addpath([parent_dir,'\grid']);
addpath([parent_dir,'\MS-PSO-pro']);
addpath([parent_dir,'\MRSM_subprograms1']);
addpath([parent_dir,'\MRSM_subprograms1_correxp']);
addpath(parent_dir); addpath(this_dir);
addpath([parent_dir,'\numeric_sample']);
fp = {'MMRGP0','MRSM','mGlobal_SUR2w_varcs02lPD_fast_piBO', ...
      'mROI_SUR2w_varcs02lPD_fast_piBO_DS','pibo_prior_weight', ...
      'mopso_adaptive_elitism_crowding','evaluate_fitness_fast', ...
      'predict_resp','error_func','prepare_roi_predict_cache'};
fprintf('===== 路径指纹 =====\n');
for k = 1:numel(fp)
    fprintf('%-40s -> %s\n', fp{k}, which(fp{k}));
end
assert(endsWith(which('MRSM'), ['MRSM_subprograms1_correxp' filesep 'MRSM.m']), 'MRSM 未解析到 correxp 覆盖目录');
assert(endsWith(which('mGlobal_SUR2w_varcs02lPD_fast_piBO'), ...
    ['numeric_sample' filesep 'mGlobal_SUR2w_varcs02lPD_fast_piBO.m']), 'C6 采样器路径错');
assert(endsWith(which('mROI_SUR2w_varcs02lPD_fast_piBO_DS'), ...
    ['numeric_sample' filesep 'mROI_SUR2w_varcs02lPD_fast_piBO_DS.m']), 'C7 采样器路径错');
assert(endsWith(which('MMRGP0'), ['Radar' filesep 'MMRGP0.m']), 'MMRGP0 未解析到 Radar 版');
if SMOKE, diary('smoke_ship_pibo_sub10_log.txt'); else, diary('ship_pibo_sub10_log.txt'); end
diary on
fprintf('===== main_ship_pibo_sub10_par 启动 %s | MATLAB %s =====\n', ...
    datestr(now,'yyyy-mm-dd HH:MM:SS'), version);
rec_file = fullfile(this_dir,'rec0405_shipbest.mat');
tv = load(rec_file, 'INPUT','OUTPUT','rsl','modelmemeroy');
INPUT = tv.INPUT;  OUTPUT = tv.OUTPUT;
meta = struct('ClassPos',cell(1,10), 'dim',zeros(1,10), ...
              'n_init',zeros(1,10), 'total_iter',zeros(1,10));
for j = 1:10
    mj = tv.modelmemeroy{j};
    meta(j).ClassPos = mj.ClassPos; meta(j).dim = mj.dim;
    meta(j).n_init = mj.n_init;     meta(j).total_iter = mj.total_iter;
end
clear tv mj
rsl = [63 55 60 85 20 17 23 43 62 78];
nS = numel(rsl);
fprintf('子集种子: rsl=%s\n', mat2str(rsl));
assert(size(INPUT,2)==meta(1).dim && size(OUTPUT,2)==numel(meta(1).ClassPos), ...
    'INPUT/OUTPUT 列数与元数据不一致');
if SMOKE
    nS = 1; seq_cap = 3;
    fprintf('===== SMOKE 模式: 1 seed x 3 步, 不落盘 =====\n');
else
    seq_cap = inf;
end
box_lo = [0 0 0 0 0.6 0 0];   box_hi = [1 1 1 1 1 1 0.4];
ew_hg = 0.06;
pi_hg = @(X) min(max(exp(-0.5*(sqrt(sum(max(box_lo-X,0).^2 + max(X-box_hi,0).^2, 2))/ew_hg).^2), 0.01), 1);
s_fn = @(X) (X(:,4) + X(:,7) - X(:,2) - X(:,6) - X(:,1) - X(:,5))/6;
c_bnd = median(s_fn(INPUT));
w_bnd = 0.08;  ew_bnd = 0.06;
pi_bnd = @(X) min(max(exp(-0.5*(max(abs(s_fn(X)-c_bnd)-w_bnd,0)/ew_bnd).^2), 0.01),  1);
pi_merged = @(X) max(pi_hg(X), pi_bnd(X));
fprintf('先验(v1): π_hg 盒 FTD_SNR∈[0.6,1]×H_dist∈[0,0.4] | π_bnd 带 |s−c|≤0.08, c=%.4f | merged=max\n', c_bnd);
arm_names    = {'C6-piBO-global','C7-piBO-ROI'};
arm_samplers = {@mGlobal_SUR2w_varcs02lPD_fast_piBO, @mROI_SUR2w_varcs02lPD_fast_piBO_DS};
nA = numel(arm_names);
nRC = 2;  ccols = 1:2;
resp_lab = {'y1=ship 响应1','y2=ship 响应2'};
tnum = 100; tsam = 70;
n_seq = meta(1).total_iter - meta(1).n_init;
pibo_beta = (meta(1).total_iter - meta(1).n_init) / 10;
fprintf('pibo_beta = (70-20)/10 = %g\n', pibo_beta);
rmse_resp = nan(n_seq, nA, nS, nRC);  rmae_resp = nan(n_seq, nA, nS, nRC);
acc_arr   = nan(n_seq, nA, nS);       t_samp   = nan(n_seq, nA, nS);
picks = cell(nA, nS);
pibo_prior_c = cell(nA, nS);  pibo_weight_c = cell(nA, nS);  pibo_n_c = cell(nA, nS);
arm_failed = false(nA, nS);  arm_errmsg = cell(nA, nS);
arm_min = nan(nA, nS);
t_all = tic;
for si = 1:nS
    cd(this_dir);
    seed_file = fullfile(this_dir, sprintf('unified_ship_pibo_sub10_seed%02d.mat', si));
    rs = rsl(si);
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
            fprintf('=== [r%d/rs=%g] 已有完成断点, 跳过 ===\n', si, rs);
            continue;
        end
        fprintf('[r%d/rs=%g] 断点不完整, 整 seed 重跑\n', si, rs);
    end
    rmse_s = nan(n_seq, nA, nRC); rmae_s = nan(n_seq, nA, nRC);
    acc_s = nan(n_seq, nA); tsamp_s = nan(n_seq, nA);
    picks_s = cell(nA,1); failed_a = false(nA,1); errmsg_a = cell(nA,1);
    arm_min_c = nan(nA,1);
    pibo_prior_s = cell(nA,1); pibo_weight_s = cell(nA,1); pibo_n_s = cell(nA,1);
    ClassPos = meta(si).ClassPos;  dim = meta(si).dim;
    model = struct();
    model.rs = rs;
    rng(rs)
    model.ns = 100;
    nrow = meta(si).n_init + model.ns;
    indeices = randperm(size(OUTPUT,1), nrow);
    trainy = OUTPUT(indeices,:); trainx = INPUT(indeices,:);
    model.candidate  = trainx(1+meta(si).n_init:end,:);
    candidatey       = trainy(1+meta(si).n_init:end,:);
    model.init_x     = trainx(1:meta(si).n_init,:);
    model.init_value = trainy(1:meta(si).n_init,:);
    indeices = randperm(size(OUTPUT,1), tnum+tsam);
    ytest = OUTPUT(indeices(1:tnum),:);  model.testset = INPUT(indeices(1:tnum),:);
    model.testsample = INPUT(indeices(1+tnum:end),:);
    model.candidate0 = model.candidate;
    model.ClassPos = ClassPos;  model.dim = dim;
    model.total_iter = meta(si).total_iter;  model.n_init = meta(si).n_init;
    model.variables = struct('ns', model.ns, 'dim', dim);
    model.MMRGP = MMRGP0(model.init_value, model.init_x, ClassPos, dim);
    assert(model.MMRGP.corrfunc == "correxp", '★MMRGP 核不是 correxp — 覆盖失败');
    model.now_value = model.init_value; model.now_x = model.init_x;
    fprintf('[r%d/rs=%g] 基座重抽完成\n', si, rs);
    for ai = 1:nA
        if failed_a(ai), continue; end
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
                [~, xind] = ismember(x_star, model.candidate0, 'rows');
                assert(xind > 0, 'x_star 不在候选池内(ismember 失败)');
                f_star = candidatey(xind,:);
                ma.now_value = [ma.now_value; f_star]; ma.now_x = [ma.now_x; x_star];
                ma.MMRGP = MMRGP0(ma.now_value, ma.now_x, ClassPos, dim);
                [ypred] = predict_resp(ma.MMRGP, model.testset);
                ev = error_func(ytest, ypred, ClassPos);
                for j = 1:nRC
                    rmae_s(i, ai, j) = ev{1}(j);
                    rmse_s(i, ai, j) = sqrt(mean((ytest(:,ccols(j)) - ypred(:,ccols(j))).^2));
                end
                acc_s(i, ai) = mean(ev{4});
                picks_s{ai}(i,:) = x_star;
                fprintf('[%s/r%d] iter %2d/%d t=%7.1fs | rmse=[%.4f %.4f] rmae=[%.4f %.4f] acc=%.4f\n', ...
                    arm_names{ai}, si, i, nsteps, tsamp_s(i,ai), ...
                    rmse_s(i,ai,1), rmse_s(i,ai,2), rmae_s(i,ai,1), rmae_s(i,ai,2), acc_s(i,ai));
            end
            arm_min_c(ai) = toc(t_arm)/60;
            fprintf('[%s/r%d] 整臂 %.1f min\n', arm_names{ai}, si, arm_min_c(ai));
            pibo_prior_s{ai} = ma.pibo_last_prior; pibo_weight_s{ai} = ma.pibo_last_weight;
            pibo_n_s{ai} = ma.pibo_last_n;
        catch err
            failed_a(ai) = true;
            errmsg_a{ai} = sprintf('%s | %s', err.identifier, err.message);
            fprintf('[%s/r%d] ★失败: %s\n', arm_names{ai}, si, errmsg_a{ai});
            for st = err.stack'
                fprintf('    at %s:%d\n', st.name, st.line);
            end
        end
    end
    rmse_resp(:,:,si,:) = rmse_s; rmae_resp(:,:,si,:) = rmae_s;
    acc_arr(:,:,si) = acc_s; t_samp(:,:,si) = tsamp_s;
    picks(:,si) = picks_s;
    pibo_prior_c(:,si) = pibo_prior_s; pibo_weight_c(:,si) = pibo_weight_s;
    pibo_n_c(:,si) = pibo_n_s;
    arm_failed(:,si) = failed_a; arm_errmsg(:,si) = errmsg_a;
    arm_min(:,si) = arm_min_c;
    if ~SMOKE
        nsteps = sum(~isnan(rmse_s(:,1)));
        save(seed_file, 'rmse_s','rmae_s','acc_s','tsamp_s','picks_s', ...
             'pibo_prior_s','pibo_weight_s','pibo_n_s', ...
             'failed_a','errmsg_a','arm_min_c','rs','nsteps','-v7');
        fprintf('=== [r%d/rs=%g] seed done(断点已存, nsteps=%d) | 总耗时 %.1f min ===\n', ...
            si, rs, nsteps, toc(t_all)/60);
    end
end
if ~SMOKE
    ckpt = fullfile(this_dir,'unified_ship_pibo_sub10_checkpoint.mat');
    save(ckpt, 'rmse_resp','rmae_resp','acc_arr','t_samp','picks', ...
         'pibo_prior_c','pibo_weight_c','pibo_n_c', ...
         'arm_failed','arm_errmsg','arm_names','rsl','n_seq','resp_lab','arm_min','-v7');
    fprintf('主循环完成: 合并 checkpoint=%s\n', ckpt);
end
nsteps_eff = min(n_seq, seq_cap);
ok = ~arm_failed & ~isnan(reshape(acc_arr(nsteps_eff,:,:), nA, nS));
fprintf('\n===== ship C6/C7 piBO·correxp核·10 seeds 末步(第 %d 步) 均值±std =====\n', nsteps_eff);
fprintf('%-22s %18s %18s %14s\n','arm','RMAE_y1','RMAE_y2','ACC');
for ai = 1:nA
    if ~any(ok(ai,:))
        msg = arm_errmsg(ai,:); msg = msg(~cellfun(@isempty,msg));
        if isempty(msg), msg = {'无数据'}; end
        fprintf('%-22s ★失败/无数据: %s\n', arm_names{ai}, strjoin(msg,' ; '));
        continue;
    end
    r1 = rmae_resp(nsteps_eff,ai,:,1); r1 = r1(ok(ai,:));
    r2 = rmae_resp(nsteps_eff,ai,:,2); r2 = r2(ok(ai,:));
    ac = acc_arr(nsteps_eff,ai,:);     ac = ac(ok(ai,:));
    fprintf('%-22s %6.4f±%.4f %14.4f±%.4f %9.4f±%.4f\n', arm_names{ai}, ...
        mean(r1), std(r1), mean(r2), std(r2), mean(ac), std(ac));
    a1 = squeeze(mean(rmae_resp(:,ai,:,1),1)); a1 = a1(ok(ai,:));
    a2 = squeeze(mean(rmae_resp(:,ai,:,2),1)); a2 = a2(ok(ai,:));
    fprintf('%-22s %6.4f±%.4f %14.4f±%.4f %14s\n', '  └ AULC', ...
        mean(a1), std(a1), mean(a2), std(a2), '');
end
fprintf('\n===== main_ship_pibo_sub10_par DONE %s | 总耗时 %.1f min =====\n', ...
    datestr(now,'yyyy-mm-dd HH:MM:SS'), toc(t_all)/60);
diary off
