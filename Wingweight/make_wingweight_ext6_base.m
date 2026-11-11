clear; close all;
t0 = tic;
[parent_dir,~,~] = fileparts(pwd);
assert(endsWith(pwd,'wingweight'), '请在 wingweight/ 目录下运行本脚本');
addpath([parent_dir,'/src']); addpath([parent_dir,'/sampling_Funciton']);
addpath([parent_dir,'/testFunc']); addpath([parent_dir,'\grid']);
addpath([parent_dir,'\MRSM_subprograms1']); addpath(parent_dir);
obj_fct = @(x)wingweight(x);
ClassPos = [1,1,2];  dim = 10;  ns = 200;  n_init = 25;
rec = load(fullfile(pwd,'0422ww25050best.mat'), 'modelmemeroy');
hj = rec.modelmemeroy{1};
assert(double(hj.rs)==198, 'cell1 rs 预期 198');
g = gen_base(198, obj_fct, ClassPos, dim, ns, n_init);
fprintf('PREFLIGHT rs=198 复算 vs 存档 cell1:\n');
assert(isequal(g.candidate0,  hj.candidate0),  'PREFLIGHT 失败: candidate0 不匹配(配方有误)');
assert(isequal(g.testsample,  hj.testsample),  'PREFLIGHT 失败: testsample 不匹配(配方有误)');
assert(isequal(g.testset,     hj.testset),     'PREFLIGHT 失败: testset 不匹配(配方有误)');
assert(isequal(g.init_x,      hj.init_x),      'PREFLIGHT 失败: init_x 不匹配(配方有误)');
iverr = max(abs(g.init_value - hj.init_value), [], 'all');
fprintf('  candidate0/testsample/testset/init_x 逐位一致 | init_value max|D|=%.2e\n', iverr);
assert(iverr < 1e-12, 'PREFLIGHT 失败: init_value 不匹配');
fprintf('PREFLIGHT PASS — 配方还原成功\n\n');
new_seeds = [11 29 47 66 84 102];
old_seeds = unique(cellfun(@(m) double(m.rs), rec.modelmemeroy));
assert(numel(intersect(new_seeds, old_seeds))==0, '新种子与既有池碰撞');
ext6base = struct('rs',{},'dim',{},'ClassPos',{},'variables',{},'n_init',{},'total_iter',{}, ...
                  'candidate0',{},'testsample',{},'testset',{},'init_x',{},'init_value',{},'my',{});
for k = 1:numel(new_seeds)
    ext6base{k} = gen_base(new_seeds(k), obj_fct, ClassPos, dim, ns, n_init);
    b = ext6base{k};
    yy = wingweight(b.testset);
    fprintf('seed %3d: candidate0=%dx%d testset=%dx%d testsample=%dx%d init=%dx%d | my=[%s] | 正类=%.4f\n', ...
        b.rs, size(b.candidate0,1), size(b.candidate0,2), size(b.testset,1), size(b.testset,2), ...
        size(b.testsample,1), size(b.testsample,2), size(b.init_x,1), size(b.init_x,2), ...
        mat2str(b.my,4), mean(yy(:,find(ClassPos==2))));
end
seeds = new_seeds;
save(fullfile(pwd,'ext6_ww_base.mat'), 'ext6base', 'seeds', '-v7');
fprintf('\n基座生成完成: ext6_ww_base.mat (%d seeds, %.1f min)\n', numel(new_seeds), toc(t0)/60);
fprintf('=== make_wingweight_ext6_base DONE OK ===\n');
function g = gen_base(rs, obj_fct, ClassPos, dim, ns, n_init)
model = struct();
model.variables.dim = dim;
model.rs = rs;
rng(rs);
model = cand_gen(model, ns, 'uniform');
model.testsample = lhsdesign(12000, dim);
model.testset    = lhsdesign(5000, dim);
model.candidate0 = model.candidate;
ytraw = obj_fct(model.testset);
my = mean(ytraw);  my(ClassPos==2) = 1;
model.init_x = lhsdesign(n_init, dim);
model.init_value = obj_fct(model.init_x) ./ my;
g = struct('rs', rs, 'dim', dim, 'ClassPos', ClassPos, ...
    'variables', struct('ns', ns, 'dim', dim), 'n_init', n_init, 'total_iter', 45, ...
    'candidate0', model.candidate0, 'testsample', model.testsample, 'testset', model.testset, ...
    'init_x', model.init_x, 'init_value', model.init_value, 'my', my);
end
