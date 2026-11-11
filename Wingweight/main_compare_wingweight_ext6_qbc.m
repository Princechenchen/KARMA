clear; close all;
nw = 4;
[parent_dir,~,~] = fileparts(pwd);
assert(endsWith(pwd,'wingweight'), '请在 wingweight/ 目录下运行本脚本');
this_dir = pwd;
addpath([parent_dir,'/src']); addpath([parent_dir,'/sampling_Funciton']);
addpath([parent_dir,'/testFunc']); addpath([parent_dir,'\grid']);
addpath([parent_dir,'\MRSM_subprograms1']); addpath([parent_dir,'\MS-PSO-pro']);
addpath([parent_dir,'\tenbar']); addpath(parent_dir);
addpath([parent_dir,'\numeric_sample']);
assert(contains(which('MMRGP0'),'wingweight'), 'MMRGP0 未解析到 wingweight/');
assert(~isempty(which('mGlobal_SUR2w_varcs02lPD_fast_QBC')), 'QBC 采样器未解析到');
B = load(fullfile(pwd,'ext6_ww_base.mat'), 'ext6base', 'seeds');
ext6base = B.ext6base;  seeds = B.seeds;  clear B
nS = numel(seeds);
fprintf('ext6 C4 补跑: seeds=%s\n', mat2str(seeds));
arm_names = {'C12-QBC-global'};
nA = 1;
base = struct();
base.variables.ns = 200; base.variables.dim = 10; base.dim = 10;
base.ClassPos = [1,1,2];
base.total_iter = 45; base.n_init = 25;
ccols = find(base.ClassPos==1);  nRC = numel(ccols);
resp_lab = {'y1=机翼重量(y1<=250)','y2=翼段指标(y2<=50)'};
n_seq = base.total_iter - base.n_init;
rmse_resp = nan(n_seq, nA, nS, nRC);  rmae_resp = nan(n_seq, nA, nS, nRC);
acc_arr = nan(n_seq, nA, nS);  t_samp = nan(n_seq, nA, nS);
picks = cell(nA, nS);
arm_failed = false(nA, nS);  arm_errmsg = cell(nA, nS);
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('Processes', nw, 'IdleTimeout', 120);
else
    fprintf('复用现有 parpool(%d worker)\n', pool.NumWorkers);
end
fprintf('===== wingweight ext6 C4 补跑启动: %d worker x %d seeds x 1 臂 x %d 步 =====\n', ...
    pool.NumWorkers, nS, n_seq);
parfor sidx = 1:nS
    cd(this_dir);
    try
        pe = pyenv;
        if ~contains(pe.Executable, 'Python39')
            pyenv('Version', 'F:\tools\Python39\python.exe');
        end
    catch
        pyenv('Version', 'F:\tools\Python39\python.exe');
    end
    pe2 = pyenv;
    fprintf('[worker s%d] python: %s\n', sidx, pe2.Executable);
    rs = double(seeds(sidx));
    acc_s    = nan(n_seq, nA);      tsamp_s  = nan(n_seq, nA);
    rmae_s   = nan(n_seq, nA, nRC); rmse_s   = nan(n_seq, nA, nRC);
    picks_s  = cell(nA, 1);
    failed_s = false(nA, 1);        errmsg_s = cell(nA, 1);
    hj = ext6base{sidx};
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
    assert(iverr < 1e-9, 'init_value 指纹不匹配(seed %d)', sidx);
    model.MMRGP      = MMRGP0(model.init_value, model.init_x, model.ClassPos, model.dim);
    model.now_value  = model.init_value; model.now_x = model.init_x;
    rng_state = rng(rs*100 + sidx);
    model.qbc_K = 8;  model.qbc_frac = 0.8;
    t_arm = tic;
    try
        for i = 1:n_seq
            tt = tic;
            [x_star, model] = mGlobal_SUR2w_varcs02lPD_fast_QBC(model);
            tsamp_s(i, 1) = toc(tt);
            f_star = wingweight(x_star);
            f_star = f_star ./ my_cell;
            model.now_value = [model.now_value; f_star]; model.now_x = [model.now_x; x_star];
            model.MMRGP = MMRGP0(model.now_value, model.now_x, model.ClassPos, model.dim);
            [ypred] = predict_resp(model.MMRGP, model.testset);
            ev = error_func(ytest_fixed, ypred, model.ClassPos);
            for j = 1:nRC
                rmae_s(i, 1, j) = ev{1}(j);
                rmse_s(i, 1, j) = sqrt(mean((ytest_fixed(:,ccols(j)) - ypred(:,ccols(j))).^2));
            end
            acc_s(i, 1) = mean(ev{4});
            picks_s{1}(i,:) = x_star;
            if mod(i,10)==0 || i==1
                fprintf('[%s/s%d-rs=%d] iter %d rmse_y=[%.4f %.4f] acc=%.4f\n', ...
                    arm_names{1}, sidx, rs, i, rmse_s(i,1,1), rmse_s(i,1,2), acc_s(i,1));
            end
        end
        fprintf('[%s/s%d-rs=%d] 整臂 %.1f min\n', arm_names{1}, sidx, rs, toc(t_arm)/60);
    catch err
        failed_s(1) = true;
        errmsg_s{1} = sprintf('%s | %s', err.identifier, err.message);
        fprintf('[%s/s%d-rs=%d] ★失败: %s\n', arm_names{1}, sidx, rs, errmsg_s{1});
    end
    rmse_resp(:,:,sidx,:) = rmse_s;   rmae_resp(:,:,sidx,:) = rmae_s;
    acc_arr(:,:,sidx)     = acc_s;    t_samp(:,:,sidx)      = tsamp_s;
    picks(:,sidx)         = picks_s;
    arm_failed(:,sidx)    = failed_s; arm_errmsg(:,sidx)    = errmsg_s;
    fprintf('=== [s%d-rs=%d] seed done ===\n', sidx, rs);
end
ckpt = fullfile(this_dir,'unified_wingweight_ext6_qbc_checkpoint.mat');
save(ckpt, 'rmse_resp','rmae_resp','acc_arr','t_samp','picks', ...
     'arm_failed','arm_errmsg','arm_names','seeds','n_seq','resp_lab','-v7');
fprintf('C4 补跑完成: %s\n', ckpt);
ok = ~arm_failed & ~isnan(reshape(acc_arr(end,:,:), nA, nS));
r = rmae_resp(:,1,:,1);  r = r(:,ok);
a = acc_arr(:,1,:);      a = a(:,ok);
t = t_samp(:,1,:);       t = t(:,ok);
fprintf('\nC4 ext6 补跑: y1 %.4f±%.4f | y2 %.4f±%.4f | ACC %.4f±%.4f | %.2f s/步\n', ...
    mean(r(end,:,1)), std(r(end,:,1)), mean(r(end,:,2)), std(r(end,:,2)), ...
    mean(a(end,:)), std(a(end,:)), mean(t(:)));
fprintf('=== main_compare_wingweight_ext6_qbc DONE OK ===\n');
