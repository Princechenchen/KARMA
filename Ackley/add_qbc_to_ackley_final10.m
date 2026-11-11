clear; close all;
F = fullfile(pwd, 'Ackley_box3_final10.mat');
Q = fullfile(pwd, 'unified_ackley_qbcg_sub10_checkpoint.mat');
snap = load(F);
Qc = load(Q);
s_acc = snap.acc_arr; s_rmae = snap.rmae_resp; s_p16acc = snap.acc_arr_p16;
s_arms = snap.arm_names; s_cells = snap.SUBSET_CELLS;
q_acc = reshape(squeeze(Qc.acc_arr(:,1,:,:)),   30, 1, 10);
q_rmae = reshape(squeeze(Qc.rmae_resp(:,1,:,:)), 30, 1, 10, 2);
q_rmse = reshape(squeeze(Qc.rmse_resp(:,1,:,:)), 30, 1, 10, 2);
q_t = reshape(squeeze(Qc.t_samp(:,1,:)),        30, 1, 10);
q_cells = double(Qc.rep_cells(:).');
assert(isequal(double(Qc.seeds(:).'), double(snap.SUBSET_SEEDS(:).')), 'QBC 种子与 final10 不符');
assert(isequal(q_cells(:).', double(snap.SUBSET_CELLS(:).')), 'QBC 格集与 final10 不符');
nA12 = numel(snap.arm_names) + 1;
acc12  = cat(2, snap.acc_arr,   q_acc);
rmae12 = cat(2, snap.rmae_resp, q_rmae);
rmse12 = cat(2, snap.rmse_resp, q_rmse);
t12    = cat(2, snap.t_samp,    q_t);
arm12 = [snap.arm_names, {'QBC-global'}];
p16_cells = double(snap.cells16(:).');
idx = zeros(1, numel(q_cells));
for k = 1:numel(q_cells)
    idx(k) = find(p16_cells == q_cells(k));
end
acc_p16  = cat(2, snap.acc_arr_p16,   nan(30, 1, 16));
rmae_p16 = cat(2, snap.rmae_resp_p16, nan(30, 1, 16, 2));
rmse_p16 = cat(2, snap.rmse_resp_p16, nan(30, 1, 16, 2));
t_p16    = cat(2, snap.t_samp_p16,    nan(30, 1, 16));
iQ = nA12;
acc_p16(:,  iQ, idx)    = q_acc;
rmae_p16(:, iQ, idx, :) = q_rmae;
rmse_p16(:, iQ, idx, :) = q_rmse;
t_p16(:,     iQ, idx)   = q_t;
arm_p16 = arm12;
S = snap;
S.arm_names = arm12;  S.acc_arr = acc12;  S.rmae_resp = rmae12;
S.rmse_resp = rmse12; S.t_samp = t12;
S.acc_arr_p16 = acc_p16; S.rmae_resp_p16 = rmae_p16;
S.rmse_resp_p16 = rmse_p16; S.t_samp_p16 = t_p16;
S.qbc_note = ['C12-QBC-global appended (finalized caliber = ', ...
    'recommended subset). Pool-based QBC AL baseline, no prior; K=8 refit committees, ', ...
    'JSD class disagreement + committee std, equal-weight z fusion; static pool with ', ...
    'labeled-mask; MMRGP0_correxp. Unselected-caliber data (old-10 cells): ', ...
    'unified_ackley_qbcg_checkpoint.mat (ACC 0.8345 tied with C8, p=1.0) — both calibers ', ...
    'must be disclosed. Paired vs C8(KRAM-EVR): RMAE sum +0.044 (1/10, p=0.020), ', ...
    'AULC sum +0.026 (3/10, ns), ACC -0.014 (3/10, ns).'];
save(F, '-struct', 'S', '-v7');
C = load(F);
assert(isequal(C.acc_arr(:, 1:11, :), s_acc), '原 11 臂 acc 被改动');
assert(isequal(C.rmae_resp(:, 1:11, :, :), s_rmae), '原 11 臂 rmae 被改动');
assert(isequal(C.acc_arr_p16(:, 1:11, :), s_p16acc), 'p16 原臂被改动');
assert(numel(C.arm_names) == 12 && strcmp(char(C.arm_names(12)), 'QBC-global'), '臂表 append 错误');
iQ = 12;
q_acc_back = C.acc_arr(:, iQ, :);
assert(all(isfinite(q_acc_back(:))), 'QBC 臂含 NaN');
assert(all(isnan(C.acc_arr_p16(:, iQ, setdiff(p16_cells, q_cells))), 'all'), 'p16 非子集应 NaN');
fprintf('FINAL10_QBC_MERGE_OK: 12 臂 = 11 原臂 + QBC-global; 原臂位级未动; 子集锚自检\n');
i3 = 11;
bacc = mean(C.acc_arr(end, i3, :)); by1 = mean(C.rmae_resp(end, i3, :, 1));
assert(abs(bacc-0.8460)<5e-4 && abs(by1-0.1483)<5e-4, 'anchor mismatch');
fprintf('FINAL10_QBC_MERGE_OK: anchor PASS (box3 ACC %.4f / y1 RMAE %.4f)\n', bacc, by1);
exit(0);
