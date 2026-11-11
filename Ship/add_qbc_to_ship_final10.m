clear; close all;
F = fullfile(pwd, 'Ship_v1box3u_final10.mat');
Q = fullfile(pwd, 'unified_ship_qbcg_checkpoint.mat');
snap = load(F);
Qc = load(Q);
s_acc = snap.acc_arr; s_rmae = snap.rmae_resp; s_p16acc = snap.acc_arr_p16;
s_arms = snap.arm_names; s_sub = snap.SUBSET_RSL;
n_seq = double(snap.n_seq);
q_acc  = reshape(squeeze(Qc.acc_arr(:,1,:,:)),   n_seq, 1, 10);
q_rmae = reshape(squeeze(Qc.rmae_resp(:,1,:,:)), n_seq, 1, 10, 2);
q_rmse = reshape(squeeze(Qc.rmse_resp(:,1,:,:)), n_seq, 1, 10, 2);
q_t    = reshape(squeeze(Qc.t_samp(:,1,:)),      n_seq, 1, 10);
q_cells = double(Qc.SUBSET_RSL(:).');
assert(isequal(q_cells(:).', double(snap.SUBSET_RSL(:).')), 'QBC 格集与 final10 不符');
iQ = numel(snap.arm_names) + 1;
acc10  = cat(2, snap.acc_arr,   q_acc);
rmae10 = cat(2, snap.rmae_resp, q_rmae);
arm10 = [snap.arm_names, {'C12-QBC-global'}];
p16_cells = [double(snap.rsl16_old(:).'), double(snap.rsl16_new(:).')];
idx = zeros(1, numel(q_cells));
for k = 1:numel(q_cells)
    idx(k) = find(p16_cells == q_cells(k));
end
acc_p16  = cat(2, snap.acc_arr_p16,   nan(n_seq, 1, 16));
rmae_p16 = cat(2, snap.rmae_resp_p16, nan(n_seq, 1, 16, 2));
acc_p16(:,  iQ, idx)    = q_acc;
rmae_p16(:, iQ, idx, :) = q_rmae;
arm_p16 = arm10;
S = snap;
S.arm_names = arm10;  S.acc_arr = acc10;  S.rmae_resp = rmae10;
S.acc_arr_p16 = acc_p16; S.rmae_resp_p16 = rmae_p16; S.arm_names_p16 = arm_p16;
S.qbc_note = ['C12-QBC-global appended (finalized caliber = recommended ', ...
    'subset SUBSET_RSL). Pool-based QBC AL baseline, no prior; K=8 refit committees, ', ...
    'JSD class disagreement + committee std, equal-weight z fusion; static pool with ', ...
    'labeled-position mask (v2; v1 value-mask invalidated on ship''s 45-unique-row pool, ', ...
    'see .aborted_valmask archive); MMRGP0 correxp; truth via row-index remap (red-line 7). ', ...
    'Unselected-caliber data: none (ship QBC ran finalized caliber only). Driver: ', ...
    'Radar/main_ship_qbcg.m.'];
save(F, '-struct', 'S', '-v7');
C = load(F);
assert(isequal(C.acc_arr(:, 1:iQ-1, :), s_acc), '原 9 臂 acc 被改动');
assert(isequal(C.rmae_resp(:, 1:iQ-1, :, :), s_rmae), '原 9 臂 rmae 被改动');
assert(isequal(C.acc_arr_p16(:, 1:iQ-1, :), s_p16acc), 'p16 原臂被改动');
assert(numel(C.arm_names) == 10 && strcmp(char(C.arm_names(10)), 'C12-QBC-global'), '臂表 append 错误');
qacc_chk = C.acc_arr(:, iQ, :);
assert(all(isfinite(qacc_chk(:))), 'QBC 臂含 NaN');
miss_pos = arrayfun(@(s) find(p16_cells == s), setdiff(p16_cells, q_cells));
assert(all(isnan(C.acc_arr_p16(:, iQ, miss_pos)), 'all'), 'p16 非子集应 NaN');
i3 = find(strcmp(C.arm_names, 'KRAM-EVR'));
assert(~isempty(i3), 'KRAM-EVR 未找到');
anch = C.anchor_subset;
by1 = mean(C.rmae_resp(end, i3, :, 1)); by2 = mean(C.rmae_resp(end, i3, :, 2));
by1a = mean(C.rmae_resp(:, i3, :, 1), 'all'); by2a = mean(C.rmae_resp(:, i3, :, 2), 'all');
bac = mean(C.acc_arr(end, i3, :)); bac2 = mean(C.acc_arr(:, i3, :), 'all');
assert(abs(by1-0.1331)<5e-4 && abs(by2-0.0537)<5e-4 && abs(by1a-0.1601)<5e-4 && ...
       abs(by2a-0.0647)<5e-4 && abs(bac-0.8720)<5e-4 && abs(bac2-0.8338)<5e-4, 'anchor mismatch');
fprintf('SHIP_FINAL10_QBC_MERGE_OK: 10 臂 = 9 原臂 + C12-QBC-global; 原臂位级未动; anchor PASS\n');
exit(0);
