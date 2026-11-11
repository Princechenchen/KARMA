clear; close all;
REP = fullfile(pwd, 'Ackley_box3_repro.mat');
Q   = fullfile(pwd, 'unified_ackley_qbcg_sub10_checkpoint.mat');
R = load(REP);
Qc = load(Q);
snap_bases16 = R.bases16;
snap_pr = R.pool_reference;
snap_rb = R.results_box3;
q.cells = double(Qc.rep_cells(:).');
q.seeds = double(Qc.seeds(:).');
q.arm_name = 'C12-QBC-global';
q.method = ['Query-by-Committee (QBC-global), pool-based AL baseline, no prior: K=8 MMRGP ', ...
    'refit on 80% class-stratified subsamples; score = z(JSD of committee class posteriors) ', ...
    '+ z(committee std of each continuous response), equal weights; static candidate0 pool, ', ...
    'labeled points masked (Settles 2009 pool protocol); kernel MMRGP0_correxp; ', ...
    'rng(double(rs)*100+cell_id); K=8, frac=0.8'];
q.acc = squeeze(Qc.acc_arr(:,1,:));
q.rmae = squeeze(Qc.rmae_resp(:,1,:,:));
q.rmse = squeeze(Qc.rmse_resp(:,1,:,:));
q.picks = Qc.picks;
q.qbc_hist = Qc.qbc_hist;
q.qbc_K_hist = Qc.qbc_K_hist;
q.qbc_terms_all = Qc.qbc_terms_all;
q.qbc_labeled_all = Qc.qbc_labeled_all;
q.summary = struct( ...
    'rmae_y1_end', mean(q.rmae(end,:,1)), 'rmae_y1_end_std', std(q.rmae(end,:,1)), ...
    'rmae_y2_end', mean(q.rmae(end,:,2)), 'rmae_y2_end_std', std(q.rmae(end,:,2)), ...
    'aulc_y1', mean(q.rmae(:,:,1)), 'aulc_y2', mean(q.rmae(:,:,2)), ...
    'acc_end', mean(q.acc(end,:)), 'acc_end_std', std(q.acc(end,:)), ...
    'acc_auc', mean(q.acc(:,:)));
i3 = 0; armsList = arrayfun(@(s) s, snap_pr.arm_names, 'UniformOutput', false);
for ii = 1:numel(armsList)
    if strcmp(char(armsList{ii}), 'C8uc-box3-COMP'), i3 = ii; end
end
idx = zeros(1, numel(q.cells));
cellsPr = double(snap_pr.cells(:).');
for k = 1:numel(q.cells)
    idx(k) = find(cellsPr == q.cells(k));
end
qe = squeeze(q.rmae(end,:,:));
qa = mean(q.rmae,1); qa = reshape(qa, numel(q.cells), 2);
ce = squeeze(snap_pr.rmae(end, i3, idx, :));
ca = mean(snap_pr.rmae(:, i3, idx, :), 1); ca = reshape(ca, numel(q.cells), 2);
q_acc_end = squeeze(q.acc(end,:));
c_acc_end = squeeze(snap_pr.acc(end, i3, idx));
dq = sum(qe,2) - sum(ce,2);
da = sum(qa,2) - sum(ca,2);
dc = q_acc_end - c_acc_end;
q.paired_vs_box3 = struct( ...
    'rmae_sum_delta_mean', mean(dq), 'rmae_sum_qbc_wins', sum(dq<0), ...
    'aulc_sum_delta_mean', mean(da), 'aulc_sum_qbc_wins', sum(da<0), ...
    'acc_delta_mean', mean(dc), 'acc_qbc_wins', sum(dc>0), ...
    'note', 'RMAE/AULC positive = QBC worse; full stats in probe_ackley_qbcg_sub10_paired.txt');
q.note = ['Finalized protocol (user 2026-10-08/09): recommended subset (8008-enum 6/6 selection). ', ...
    'Sensitivity caveat: on the UNSELECTED old-10 cells (seeds [172 136 34 46 77 160 91 158 180 1]) ', ...
    'QBC ACC 0.8345 was statistically TIED with box3 (p=1.0) and ranked 1st on average — both ', ...
    'calibers must be disclosed together in the paper (selection bias transparency). ', ...
    'Old-10 data: unified_ackley_qbcg_checkpoint.mat; probes: probe_ackley_qbcg_sub10_paired.py / ', ...
    'probe_ackley_allarms_sub10_table.py; driver: main_compare_ackley_qbcg.m (RUN_MODE=sub10). ', ...
    'Committee class-disagreement channel alive 293/300 steps.'];
q.merged_date = '2026-10-09';
R.qbc_sub10 = q;
save(REP, '-struct', 'R', '-v7');
C = load(REP);
assert(isequal(C.bases16, snap_bases16), 'bases16 被改动');
assert(isequal(double(C.pool_reference.acc(:)), double(snap_pr.acc(:))), 'pool_reference 被改动');
assert(isequal(double(C.results_box3.acc(:)), double(snap_rb.acc(:))), 'results_box3 被改动');
assert(isequal(C.subset_selection.sel_cells, R.subset_selection.sel_cells), 'subset_selection 被改动');
idx3 = 0; armsC = arrayfun(@(s) s, C.pool_reference.arm_names, 'UniformOutput', false);
for ii = 1:numel(armsC)
    if strcmp(char(armsC{ii}), 'C8uc-box3-COMP'), idx3 = ii; end
end
cellsC = double(C.pool_reference.cells(:).');
for k = 1:numel(q.cells), idx(k) = find(cellsC == q.cells(k)); end
bacc = mean(squeeze(C.pool_reference.acc(end, idx3, idx)));
by1  = mean(squeeze(C.pool_reference.rmae(end, idx3, idx, 1)));
assert(abs(bacc-0.8460)<5e-4 && abs(by1-0.1483)<5e-4, 'anchor mismatch');
assert(all(isfinite(C.qbc_sub10.acc(:))), 'qbc_sub10.acc 含 NaN');
assert(numel(C.qbc_sub10.cells) == 10, 'qbc_sub10 格数不对');
fprintf('QBC_MERGE_OK: qbc_sub10 已并入 %s\n  原字段位级未动; anchor PASS; 格集 %s\n', ...
    REP, mat2str(q.cells));
exit(0);
