clear; close all;
F = fullfile(pwd, 'Wingweight_box3u_final10.mat');
Q = fullfile(pwd, 'unified_wingweight_pibo_sub10_checkpoint.mat');
snap = load(F);
Qc = load(Q);
s_acc = snap.acc_arr; s_rmae = snap.rmae_resp; s_arms = snap.arm_names;
s_sub = snap.SUBSET_SEEDS; s_acc22 = snap.acc_arr_p22; s_qacc = snap.qbc_acc_arr_sub10;
assert(isequal(double(Qc.rsl(:).'), double(snap.SUBSET_SEEDS(:).')), ...
    'piBO 种子序与 final10 SUBSET_SEEDS 不符');
n_seq = double(snap.n_seq);
assert(size(Qc.acc_arr,1)==n_seq && size(Qc.acc_arr,2)==2 && size(Qc.acc_arr,3)==10, ...
    'piBO acc_arr 维度异常');
assert(~any(Qc.arm_failed(:)), 'piBO 臂存在失败标记');
S = snap;
S.pibo_arm_names = {'C6-piBO-global', 'C7-piBO-ROI'};
S.pibo_acc_arr_sub10   = Qc.acc_arr;
S.pibo_rmae_resp_sub10 = Qc.rmae_resp;
S.pibo_rmse_resp_sub10 = Qc.rmse_resp;
S.pibo_t_samp_sub10    = Qc.t_samp;
S.pibo_note = ['C6/C7 piBO arms appended (subset-only; no ', ...
    'runs). Static merged prior max(pi_hg,pi_bnd) from src/pi_wingweight_prior.m (v1, ', ...
    'same geometry as C8 box3u), full strength, no arbitration; pibo_beta=(45-25)/10=2 ', ...
    '(piBO paper rule). C6 = numeric_sample/mGlobal_SUR2w_varcs02lPD_fast_piBO (global ', ...
    'pool), C7 = numeric_sample/mROI_SUR2w_varcs02lPD_fast_piBO (data ROI, non-DS). ', ...
    'MMRGP0 = wingweight local (corrgauss); rng(rs*100+si) with si = original base ', ...
    'position (ext6#2-6/ext4#4/ext4b#1/ext4c#3,4/ext4d#2) - init fingerprint max|D|=0. ', ...
    'Run: main_compare_wingweight_pibo_sub10.m via run_ww_pibo_chain.sh (ALLDONE ', ...
    '2026-11-04 11:42, 10/10 seeds, 0 failures).'];
save(F, '-struct', 'S', '-v7');
C = load(F);
assert(isequal(C.acc_arr, s_acc), '主 acc_arr 被改动');
assert(isequal(C.rmae_resp, s_rmae), '主 rmae_resp 被改动');
assert(isequal(C.arm_names, s_arms) && numel(C.arm_names)==10, '主臂表被改动');
assert(isequal(C.acc_arr_p22, s_acc22), 'p22 池被改动');
assert(isequal(C.qbc_acc_arr_sub10, s_qacc), 'qbc 字段被改动');
assert(isequal(double(C.SUBSET_SEEDS(:).'), double(s_sub(:).')), 'SUBSET_SEEDS 被改动');
i8 = find(strcmp(C.arm_names, 'KRAM-EVR'));
assert(numel(i8)==1, 'KRAM-EVR 臂未找到');
by1  = mean(C.rmae_resp(end, i8, :, 1));   by2  = mean(C.rmae_resp(end, i8, :, 2));
by1a = mean(C.rmae_resp(:, i8, :, 1), 'all'); by2a = mean(C.rmae_resp(:, i8, :, 2), 'all');
bac  = mean(C.acc_arr(end, i8, :));       bac2 = mean(C.acc_arr(:, i8, :), 'all');
assert(abs(by1-0.0077)<5e-4 && abs(by2-0.0107)<5e-4 && abs(bac-0.8727)<5e-4 && ...
       abs(bac2-0.8327)<5e-4 && abs(by1a-0.0095)<5e-4 && abs(by2a-0.0138)<5e-4, ...
    'KRAM-EVR anchor mismatch');
p_acc = C.pibo_acc_arr_sub10;  p_rmae = C.pibo_rmae_resp_sub10;
assert(all(isfinite(p_acc(:))) && all(isfinite(p_rmae(:))), 'pibo 字段含 NaN');
c6y1 = mean(p_rmae(end,1,:,1)); c6y2 = mean(p_rmae(end,1,:,2)); c6ac = mean(p_acc(end,1,:));
c7y1 = mean(p_rmae(end,2,:,1)); c7y2 = mean(p_rmae(end,2,:,2)); c7ac = mean(p_acc(end,2,:));
assert(abs(c6y1-0.0075)<5e-4 && abs(c6y2-0.0107)<5e-4 && abs(c6ac-0.8416)<5e-4, ...
    'C6 锚未命中(与 13 方法表不符)');
assert(abs(c7y1-0.0076)<5e-4 && abs(c7y2-0.0106)<5e-4 && abs(c7ac-0.8402)<5e-4, ...
    'C7 锚未命中(与 13 方法表不符)');
fprintf('WW_FINAL10_PIBO_MERGE_OK: 原字段位级未动; KRAM-EVR 锚 PASS; C6/C7 锚 PASS\n');
fprintf('C6: y1=%.4f y2=%.4f ACC=%.4f | C7: y1=%.4f y2=%.4f ACC=%.4f\n', ...
    c6y1, c6y2, c6ac, c7y1, c7y2, c7ac);
exit(0);
