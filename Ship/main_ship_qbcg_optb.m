clear; close all; clc;
SMOKE = ~isempty(getenv('SHIP_QBC_OPTB_SMOKE'));
QBC_K = 8;  QBC_FRAC = 0.8;
[parent_dir,~,~] = fileparts(pwd);
assert(endsWith(pwd,'Radar'), '请在 Radar/ 目录下运行本脚本');
this_dir = pwd;
addpath([parent_dir,'/src']); addpath([parent_dir,'/sampling_Funciton']);
addpath([parent_dir,'/testFunc']); addpath([parent_dir,'\grid']);
addpath([parent_dir,'\MS-PSO-pro']); addpath([parent_dir,'\MRSM_subprograms1']);
addpath([parent_dir,'\MRSM_subprograms1_correxp']);
addpath(parent_dir); addpath(this_dir);
addpath([parent_dir,'\numeric_sample']);
assert(endsWith(which('MRSM'), ...
    ['MRSM_subprograms1_correxp' filesep 'MRSM.m']), 'MRSM 未解析到 correxp 覆盖目录');
assert(endsWith(which('MMRGP0'), ['Radar' filesep 'MMRGP0.m']), 'MMRGP0 未解析到 Radar 版');
assert(endsWith(which('mGlobal_SUR2w_varcs02lPD_fast_QBC'), ...
    ['numeric_sample' filesep 'mGlobal_SUR2w_varcs02lPD_fast_QBC.m']), 'QBC 采样器未解析到 numeric_sample');
assert(endsWith(which('error_func'), ['src' filesep 'error_func.m']), 'error_func 未解析到 src');
progf = fullfile(this_dir, 'ship_qbcg_optb_prog.txt');
if SMOKE, progf = fullfile(this_dir, 'ship_qbcg_optb_smoke_prog.txt'); end
fp = fopen(progf,'w'); fprintf(fp,'ship QBC-global start %s (SMOKE=%d, K=%d, frac=%.2f)\n', ...
    datetime('now'), SMOKE, QBC_K, QBC_FRAC); fclose(fp);
fin = load(fullfile(this_dir,'Ship_v1box3u_final10.mat'), 'INPUT','OUTPUT','ClassPos','SUBSET_RSL');
INPUT = fin.INPUT; OUTPUT = fin.OUTPUT;
ClassPos = double(fin.ClassPos(:).'); dim = size(INPUT,2);
assert(numel(ClassPos)==3 && size(OUTPUT,2)==3, 'ClassPos/OUTPUT 口径异常');
SUBSET_RSL = [5 43];
nS = numel(SUBSET_RSL);
if SMOKE, nS = 1; end
fprintf('INPUT=%dx%d | ClassPos=%s | SUBSET_RSL=%s\n', size(INPUT,1), size(INPUT,2), ...
    mat2str(ClassPos), mat2str(SUBSET_RSL));
nRC = 2;  ccols = 1:2;
tnum = 100; tsam = 70;
n_init = 20; total_iter = 70; n_seq = total_iter - n_init;
if SMOKE, n_run = 3; else, n_run = n_seq; end
nA = 1;
rmse_resp = nan(n_run, nA, nS, nRC);  rmae_resp = nan(n_run, nA, nS, nRC);
acc_arr = nan(n_run, nA, nS);  t_samp = nan(n_run, nA, nS);
picks = cell(nA, nS);  pick_idx = cell(nA, nS);
qbc_hist = cell(nA, nS);  qbc_K_hist = cell(nA, nS);
qbc_terms_all = cell(nA, nS);  qbc_labeled_all = cell(nA, nS);
arm_failed = false(nA, nS);  arm_errmsg = cell(nA, nS);
ckpt = fullfile(this_dir, 'unified_ship_qbcg_optb_checkpoint.mat');
start_si = 1;
if ~SMOKE && isfile(ckpt)
    C = load(ckpt);
    needf = {'rmse_resp','rmae_resp','acc_arr','t_samp','picks','pick_idx','qbc_hist', ...
             'qbc_K_hist','qbc_terms_all','qbc_labeled_all','arm_failed','arm_errmsg','SUBSET_RSL'};
    if all(isfield(C,needf)) && numel(C.SUBSET_RSL)<=nS && isequal(double(C.SUBSET_RSL(:).'), SUBSET_RSL(1:numel(C.SUBSET_RSL)))
        k = numel(C.SUBSET_RSL);
        doneSi = find(~all(isnan(reshape(C.acc_arr(end,:,:), nA, k)), 1));
        if ~isempty(doneSi)
            rmse_resp(:,:,1:k,:) = C.rmse_resp;  rmae_resp(:,:,1:k,:) = C.rmae_resp;
            acc_arr(:,:,1:k) = C.acc_arr;  t_samp(:,:,1:k) = C.t_samp;
            picks(:,1:k) = C.picks;  pick_idx(:,1:k) = C.pick_idx;
            qbc_hist(:,1:k) = C.qbc_hist;  qbc_K_hist(:,1:k) = C.qbc_K_hist;
            qbc_terms_all(:,1:k) = C.qbc_terms_all;  qbc_labeled_all(:,1:k) = C.qbc_labeled_all;
            arm_failed(:,1:k) = C.arm_failed; arm_errmsg(:,1:k) = C.arm_errmsg;
            start_si = max(doneSi) + 1;
            fprintf('续跑: 1-%d 已完成, 从 %d (rs=%g) 开始\n', max(doneSi), start_si, SUBSET_RSL(start_si));
        end
    end
end
for si = start_si:nS
    rs = SUBSET_RSL(si);
    model = struct();
    model.rs = rs;
    rng(rs)
    model.ns = 100;
    nrow = n_init + model.ns;
    indeices = randperm(size(OUTPUT,1), nrow);
    trainy = OUTPUT(indeices,:); trainx = INPUT(indeices,:);
    model.candidate  = trainx(n_init+1:end,:);
    candidatey       = trainy(n_init+1:end,:);
    model.init_x     = trainx(1:n_init,:);
    model.init_value = trainy(1:n_init,:);
    indeices = randperm(size(OUTPUT,1), tnum+tsam);
    ytest = OUTPUT(indeices(1:tnum),:);  model.testset = INPUT(indeices(1:tnum),:);
    model.testsample = INPUT(indeices(1+tnum:end),:);
    model.candidate0 = model.candidate;
    model.ClassPos = ClassPos;  model.dim = dim;
    model.total_iter = total_iter;  model.n_init = n_init;
    model.variables = struct('ns', model.ns, 'dim', dim);
    model.MMRGP = MMRGP0(model.init_value, model.init_x, ClassPos, dim);
    assert(model.MMRGP.corrfunc == "correxp", '★MMRGP 核不是 correxp — 覆盖失败');
    model.now_value = model.init_value; model.now_x = model.init_x;
    fp = fopen(progf,'a'); fprintf(fp,'seed r%d/rs=%g base ready, split fingerprint sum(init rows)=%d\n', ...
        si, rs, sum(indeices(1:n_init))); fclose(fp);
    rng_state = rng;
    ai = 1;
    t_arm = tic;
    try
        rng(rng_state);
        ma = model;
        ma.qbc_K = QBC_K;  ma.qbc_frac = QBC_FRAC;
        fp = fopen(progf,'a'); fprintf(fp,'arm C12-QBC-global start (r%d/rs=%g)\n', si, rs); fclose(fp);
        for i = 1:n_run
            tt = tic;
            [x_star, ma] = mGlobal_SUR2w_varcs02lPD_fast_QBC(ma);
            t_samp(i, ai, si) = toc(tt);
            xind = ma.qbc_pick_index;
            assert(xind >= 1 && xind <= size(model.candidate0,1), 'pick_index 越界');
            assert(max(abs(x_star - model.candidate0(xind,:))) < 1e-12, 'pick 与索引行不一致');
            f_star = candidatey(xind,:);
            ma.now_value = [ma.now_value; f_star]; ma.now_x = [ma.now_x; x_star];
            ma.MMRGP = MMRGP0(ma.now_value, ma.now_x, ClassPos, dim);
            [ypred] = predict_resp(ma.MMRGP, model.testset);
            ev = error_func(ytest, ypred, ClassPos);
            for j = 1:nRC
                rmae_resp(i, ai, si, j) = ev{1}(j);
                rmse_resp(i, ai, si, j) = sqrt(mean((ytest(:,ccols(j)) - ypred(:,ccols(j))).^2));
            end
            acc_arr(i, ai, si) = mean(ev{4});
            picks{ai, si}(i,:) = x_star;
            pick_idx{ai, si}(i,1) = xind;
            qbc_hist{ai, si} = ma.qbc_hist;
            qbc_K_hist{ai, si} = ma.qbc_K_hist;
            qbc_terms_all{ai, si}{i} = ma.qbc_terms;
            qbc_labeled_all{ai, si}{i} = ma.qbc_labeled;
            if mod(i,5)==0 || i==1
                fprintf('[C12-QBC-global/r%d] iter %2d/%d t=%6.1fs | rmae=[%.4f %.4f] acc=%.4f\n', ...
                    si, i, n_run, t_samp(i,ai,si), rmae_resp(i,ai,si,1), rmae_resp(i,ai,si,2), acc_arr(i,ai,si));
            end
            fp = fopen(progf,'a'); fprintf(fp,'r%d-c(rs=%g) step %d ok (%.1f s)\n', si, rs, i, t_samp(i,ai,si)); fclose(fp);
        end
        fprintf('[C12-QBC-global/r%d] 整臂 %.1f min (均值 %.1f s/步)\n', si, toc(t_arm)/60, ...
            sum(t_samp(:,ai,si),'omitnan')/sum(~isnan(t_samp(:,ai,si))));
        fp = fopen(progf,'a'); fprintf(fp,'arm done (r%d)\n', si); fclose(fp);
    catch err
        arm_failed(ai, si) = true;
        arm_errmsg{ai, si} = sprintf('%s | %s', err.identifier, err.message);
        fprintf('[C12-QBC-global/r%d] ★失败: %s\n', si, arm_errmsg{ai,si});
        for st = err.stack'
            fprintf('    at %s:%d\n', st.name, st.line);
        end
        fp = fopen(progf,'a'); fprintf(fp,'arm FAILED (r%d): %s\n', si, err.message); fclose(fp);
    end
    if ~SMOKE
        arm_names = {'C12-QBC-global'};
        save(ckpt, 'rmse_resp','rmae_resp','acc_arr','t_samp','picks','pick_idx','qbc_hist', ...
             'qbc_K_hist','qbc_terms_all','qbc_labeled_all','arm_failed','arm_errmsg','arm_names', ...
             'SUBSET_RSL','n_seq','n_init','total_iter','QBC_K','QBC_FRAC','-v7');
    end
    fprintf('=== seed %d/%d (rs=%g) done ===\n', si, nS, rs);
    fp = fopen(progf,'a'); fprintf(fp,'seed r%d done\n', si); fclose(fp);
end
ok = ~arm_failed & ~isnan(reshape(acc_arr(end,:,:), nA, nS));
if any(ok(:))
    fprintf('\n===== C12-QBC-global 末步 RMAE 均值±std + AULC, seeds=%s =====\n', mat2str(SUBSET_RSL));
    for j = 1:nRC
        r = squeeze(rmae_resp(:,1,:,j)); r = r(:, ok(1,:));
        fprintf('y%d: %6.4f±%.4f  AULC %8.4f\n', j, mean(r(end,:)), std(r(end,:)), mean(mean(r,1)));
    end
    a = squeeze(acc_arr(:,1,:)); a = a(:, ok(1,:));
    fprintf('ACC: %6.4f±%.4f  ACC-AUC %8.4f\n', mean(a(end,:)), std(a(end,:)), mean(mean(a,1)));
    fprintf('采样均值 %.1f s/步\n', mean(mean(t_samp(:,1,:),1,'omitnan')));
end
fprintf('\n主循环完成: checkpoint=%s\n 对照 = Ship_v1box3u_final10.mat p16 口径\n', ckpt);
fp = fopen(progf,'a'); fprintf(fp,'ALLDONE\n'); fclose(fp);
exit(0);
