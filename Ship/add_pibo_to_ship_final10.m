clear; close all;
F = fullfile(pwd, 'Ship_v1box3u_final10.mat');
Q1 = load(fullfile(pwd, 'unified_ship_pibo_sub10_checkpoint.mat'));
Q2 = load(fullfile(pwd, 'unified_ship_pibo_ext6_checkpoint.mat'));
snap = load(F);
s_acc = snap.acc_arr; s_rmae = snap.rmae_resp;
s_acc16 = snap.acc_arr_p16; s_rmae16 = snap.rmae_resp_p16;
s_arms = snap.arm_names; s_arms16 = snap.arm_names_p16;
s_sub = double(snap.SUBSET_RSL(:).');
r1 = double(Q1.rsl(:).');
r2 = double(Q2.rsl(:).');
n_seq = double(snap.n_seq);
q_acc  = nan(n_seq, 2, 10);
q_rmae = nan(n_seq, 2, 10, 2);
for k = 1:10
    [which, pos] = local_src(s_sub(k), r1, r2);
    if which == 1
        q_acc(:, :, k)   = reshape(squeeze(Q1.acc_arr(:, :, pos)),   n_seq, 2);
        q_rmae(:, :, k, :) = reshape(squeeze(Q1.rmae_resp(:, :, pos, :)), n_seq, 2, 2);
    else
        q_acc(:, :, k)   = reshape(squeeze(Q2.acc_arr(:, :, pos)),   n_seq, 2);
        q_rmae(:, :, k, :) = reshape(squeeze(Q2.rmae_resp(:, :, pos, :)), n_seq, 2, 2);
    end
end
assert(all(isfinite(q_acc(:))) && all(isfinite(q_rmae(:))), '子集装配含 NaN');
p16_cells = [double(snap.rsl16_old(:).'), double(snap.rsl16_new(:).')];
q_acc16  = nan(n_seq, 2, 16);
q_rmae16 = nan(n_seq, 2, 16, 2);
for k = 1:16
    [which, pos] = local_src(p16_cells(k), r1, r2);
    assert(which > 0, 'p16 种子 %g 无 piBO 数据', p16_cells(k));
    if which == 1
        q_acc16(:, :, k)   = reshape(squeeze(Q1.acc_arr(:, :, pos)),   n_seq, 2);
        q_rmae16(:, :, k, :) = reshape(squeeze(Q1.rmae_resp(:, :, pos, :)), n_seq, 2, 2);
    else
        q_acc16(:, :, k)   = reshape(squeeze(Q2.acc_arr(:, :, pos)),   n_seq, 2);
        q_rmae16(:, :, k, :) = reshape(squeeze(Q2.rmae_resp(:, :, pos, :)), n_seq, 2, 2);
    end
end
assert(all(isfinite(q_acc16(:))) && all(isfinite(q_rmae16(:))), 'p16 装配含 NaN');
S = snap;
S.acc_arr   = cat(2, snap.acc_arr,   q_acc);
S.rmae_resp = cat(2, snap.rmae_resp, q_rmae);
S.acc_arr_p16   = cat(2, snap.acc_arr_p16,   q_acc16);
S.rmae_resp_p16 = cat(2, snap.rmae_resp_p16, q_rmae16);
S.arm_names    = [snap.arm_names,    {'C6-piBO-global'}, {'C7-piBO-ROI'}];
S.arm_names_p16 = S.arm_names;
S.pibo_note = ['C6/C7 piBO arms appended from the archived runs ', ...
    '(unified_ship_pibo_sub10_checkpoint.mat + unified_ship_pibo_ext6_checkpoint.mat; ', ...
    'merged coverage = full 16-pool, subset columns follow SUBSET_RSL; 74/16 from ext6). ', ...
    'Static merged prior max(pi_hg,pi_bnd) of the same v1 geometry as C8 v1box3u, full ', ...
    'strength, no arbitration; pibo_beta=(70-20)/10=5 (piBO paper rule). C6 = global-pool ', ...
    'M_data*pi^(beta/n); C7 = data-ROI + in-ROI M_data*pi^(beta/n) (_DS variant, pool ', ...
    'pick + row-index remap, red-line 7). MMRGP0 correxp (driver runtime asserts). ', ...
    'Drivers: Radar/main_ship_pibo_sub10_par.m / main_ship_pibo_ext6_par.m.'];
save(F, '-struct', 'S', '-v7');
C = load(F);
n0 = numel(s_arms);
assert(isequal(C.acc_arr(:, 1:n0, :), s_acc), '原 10 臂 acc 被改动');
assert(isequal(C.rmae_resp(:, 1:n0, :, :), s_rmae), '原 10 臂 rmae 被改动');
assert(isequal(C.acc_arr_p16(:, 1:n0, :), s_acc16), 'p16 原臂被改动');
assert(numel(C.arm_names) == 12, '臂表 append 错误');
i8 = find(strcmp(C.arm_names, 'KRAM-EVR'));
by1 = mean(C.rmae_resp(end, i8, :, 1)); by2 = mean(C.rmae_resp(end, i8, :, 2));
by1a = mean(C.rmae_resp(:, i8, :, 1), 'all'); by2a = mean(C.rmae_resp(:, i8, :, 2), 'all');
bac = mean(C.acc_arr(end, i8, :)); bac2 = mean(C.acc_arr(:, i8, :), 'all');
assert(abs(by1-0.1331)<5e-4 && abs(by2-0.0537)<5e-4 && abs(by1a-0.1601)<5e-4 && ...
       abs(by2a-0.0647)<5e-4 && abs(bac-0.8720)<5e-4 && abs(bac2-0.8338)<5e-4, 'KRAM-EVR 锚未命中');
i6 = find(strcmp(C.arm_names, 'C6-piBO-global')); i7 = find(strcmp(C.arm_names, 'C7-piBO-ROI'));
for ii = [i6, i7]
    v = [mean(C.rmae_resp(end,ii,:,1)), mean(C.rmae_resp(end,ii,:,2)), ...
         mean(C.rmae_resp(:,ii,:,1),'all'), mean(C.rmae_resp(:,ii,:,2),'all'), ...
         mean(C.acc_arr(end,ii,:)), mean(C.acc_arr(:,ii,:),'all')];
    if ii == i6, ref = [0.1354 0.0545 0.1625 0.0647 0.8680 0.8434]; end
    if ii == i7, ref = [0.1348 0.0545 0.1641 0.0647 0.8490 0.8390]; end
    assert(all(abs(v-ref) < 5e-4), 'C6/C7 锚未命中(与 12 方法表不符)');
    fprintf('arm %d: %s\n', ii, mat2str(v, 4));
end
fprintf('SHIP_FINAL10_PIBO_MERGE_OK: 12 臂 = 10 原臂 + C6/C7; 原臂位级未动; 全锚 PASS\n');
function [which, pos] = local_src(s, r1, r2)
which = 0; pos = 0;
f = find(r1 == s, 1);
if ~isempty(f), which = 1; pos = f; return; end
f = find(r2 == s, 1);
if ~isempty(f), which = 2; pos = f; return; end
end
