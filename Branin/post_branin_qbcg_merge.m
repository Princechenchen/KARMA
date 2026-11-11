clear; close all;
[parent_dir,~,~] = fileparts(pwd);
addpath(parent_dir);
P = load(fullfile(pwd,'unified_branin_pool16.mat'));
Q = load(fullfile(pwd,'unified_branin_qbcg_checkpoint.mat'));
B1 = load(fullfile(pwd,'unified_branin_box3d_checkpoint.mat'));
B2 = load(fullfile(pwd,'unified_branin_box3d_ext6_checkpoint.mat'));
cells16 = double(P.cells16(:).');
seeds16 = double(P.seeds16(:).');
assert(isequal(cells16, 1:16), 'cells16 应为 1..16');
nA_old = numel(P.arm_names);
nA = nA_old + 2;
[n_seq, ~, nC, nRC] = size(P.rmae14);
assert(n_seq==20 && nC==16 && nRC==2, 'pool16 形状不符');
b_cells = [double(B1.SUBSET_CELLS(:).'), double(B2.SUBSET_CELLS(:).')];
b_seeds = [double(B1.seeds(:).'),     double(B2.seeds(:).')];
rmae_b = cat(2, squeeze(B1.rmae_resp), squeeze(B2.rmae_resp));
rmse_b = cat(2, squeeze(B1.rmse_resp), squeeze(B2.rmse_resp));
acc_b  = cat(2, squeeze(B1.acc_arr),  squeeze(B2.acc_arr));
t_b    = cat(2, squeeze(B1.t_samp),   squeeze(B2.t_samp));
b_idx = zeros(1,16);
for c = 1:16
    m = find(b_cells == cells16(c));
    assert(numel(m)==1 && b_seeds(m)==seeds16(c), 'box3d cell %d 匹配失败', c);
    b_idx(c) = m;
end
q_cells = double(Q.SUBSET_CELLS(:).');
q_seeds = double(Q.seeds(:).');
nQ = numel(q_cells);
rmae_q = squeeze(Q.rmae_resp(:,1,:,:));
rmse_q = squeeze(Q.rmse_resp(:,1,:,:));
acc_q  = squeeze(Q.acc_arr(:,1,:));
t_q    = squeeze(Q.t_samp(:,1,:));
assert(isequal(size(rmae_q), [n_seq, nQ, nRC]), 'QBC rmae 形状不符');
acc_all  = cat(2, P.acc14,  nan(n_seq, 2, nC));
rmae_all = cat(2, P.rmae14, nan(n_seq, 2, nC, nRC));
rmse_all = cat(2, P.rmse14, nan(n_seq, 2, nC, nRC));
t_all    = cat(2, P.t14,    nan(n_seq, 2, nC));
arm_failed_all = cat(1, double(P.arm_failed16), ones(2, nC));
picks_all = cell(nA, nC);  picks_all(1:nA_old, 1:nC) = P.picks14;
alpha_all = cell(nA, nC);  alpha_all(1:nA_old, 1:nC) = P.alpha16;
cons_all  = cell(nA, nC);  cons_all(1:nA_old, 1:nC)  = P.cons16;
b_a = nA_old + 1;   qbc_a = nA_old + 2;
for c = 1:16
    acc_all(:, b_a, c) = acc_b(:, b_idx(c));
    rmae_all(:, b_a, c, :) = rmae_b(:, b_idx(c), :);
    rmse_all(:, b_a, c, :) = rmse_b(:, b_idx(c), :);
    t_all(:, b_a, c) = t_b(:, b_idx(c));
    arm_failed_all(b_a, c) = 0;
    picks_all{b_a, c} = [];
end
for k = 1:numel(B1.SUBSET_CELLS)
    picks_all{b_a, find(cells16 == B1.SUBSET_CELLS(k))} = B1.picks{k};
end
for k = 1:numel(B2.SUBSET_CELLS)
    picks_all{b_a, find(cells16 == B2.SUBSET_CELLS(k))} = B2.picks{k};
end
qbc_idx = qbc_a;
for k = 1:nQ
    c = q_cells(k);
    assert(seeds16(c) == q_seeds(k), 'cell %d 种子不符: %d vs %d', c, seeds16(c), q_seeds(k));
    acc_all(:, qbc_a, c)   = acc_q(:, k);
    rmae_all(:, qbc_a, c, :) = rmae_q(:, k, :);
    rmse_all(:, qbc_a, c, :) = rmse_q(:, k, :);
    t_all(:, qbc_a, c)     = t_q(:, k);
    arm_failed_all(qbc_a, c) = 0;
    picks_all{qbc_a, c} = Q.picks{k};
end
arm_names_all = [P.arm_names, {'C8uc-box3d-COMP'}, {'C12-QBC-global'}];
impl_ver = P.impl_ver;
qbc_info = struct('source', 'unified_branin_qbcg_checkpoint.mat', ...
    'subset_cells', mat2str(q_cells), 'subset_seeds', mat2str(q_seeds), ...
    'method', 'QBC-global: K=8 委员会(80%无放回类别分层子采样重拟合 MMRGP), 分歧=JS软类分歧+连续跨委员std, z等权; 静态池选点, 已标注点屏蔽; 无先验', ...
    'seed_caveat', 'rs 字段 uint8 饱和: 10 格实际共用 rng(255), 格间差异来自初始基座; cells 8/9(rs=30 对,基座全同)轨迹位级全同属预期; 修复=double cast, 用户决定不重跑', ...
    'merged_date', '2026-10-08');
save(fullfile(pwd,'unified_branin_pool16_qbc.mat'), 'arm_names_all', 'acc_all', 'rmae_all', 'rmse_all', ...
    't_all', 'picks_all', 'arm_failed_all', 'alpha_all', 'cons_all', 'cells16', 'seeds16', 'impl_ver', ...
    'n_seq', 'qbc_info', '-v7');
Chk = load(fullfile(pwd,'unified_branin_pool16_qbc.mat'));
assert(isequal(Chk.acc_all(:, 1:nA_old, :), P.acc14), '旧臂 acc 被改动');
assert(isequal(Chk.rmae_all(:, 1:nA_old, :, :), P.rmae14), '旧臂 rmae 被改动');
assert(all(isfinite(Chk.acc_all(:, b_a, :)), 'all'), 'box3d 应全格有数据');
assert(all(isfinite(Chk.acc_all(:, qbc_a, q_cells)), 'all'), 'QBC 子集含 NaN');
assert(all(isnan(Chk.acc_all(:, qbc_a, setdiff(1:16, q_cells))), 'all'), 'QBC 非子集应全 NaN');
sc = [2,3,6,7,8,9,10,14,1,16];
bd_acc = mean(Chk.acc_all(end, b_a, sc));  bd_y1 = mean(squeeze(Chk.rmae_all(end, b_a, sc, 1)));
bd_auc = mean(mean(Chk.acc_all(:, b_a, sc), 1));
assert(abs(bd_acc-0.9333)<5e-4 && abs(bd_y1-0.2233)<5e-4 && abs(bd_auc-0.9294)<5e-4, 'box3d 锚值不符');
qb_acc = mean(Chk.acc_all(end, qbc_a, sc)); qb_y1 = mean(squeeze(Chk.rmae_all(end, qbc_a, sc, 1)));
qb_auc = mean(mean(Chk.acc_all(:, qbc_a, sc), 1));
assert(abs(qb_acc-0.9250)<5e-4 && abs(qb_y1-0.2957)<5e-4 && abs(qb_auc-0.9163)<5e-4, 'QBC 锚值不符');
fprintf('MERGE_OK: 14 臂 = 12 pool + box3d(全格) + QBC(子集); 旧臂位级未动; 锚值断言过\n');
