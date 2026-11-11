clear; close all;
[parent_dir,~,~] = fileparts(pwd);
assert(endsWith(pwd,'wingweight'), '请在 wingweight/ 目录下运行本脚本');
addpath([parent_dir,'/src']); addpath([parent_dir,'/sampling_Funciton']);
addpath([parent_dir,'/testFunc']); addpath([parent_dir,'\grid']);
addpath([parent_dir,'\MRSM_subprograms1']); addpath(parent_dir);
want = [11, 29, 47, 66, 84, 102, 74, 110, 133, 188];
srcs = {'ext6_ww_base.mat','ext6base'; ...
        'ext4_ww_base.mat','ext4base'; ...
        'ext4b_ww_base.mat','ext4base'; ...
        'ext4c_ww_base.mat','ext4base'};
subset10base = cell(1, numel(want)); got = zeros(1, numel(want));
for k = 1:size(srcs,1)
    B = load(fullfile(pwd, srcs{k,1}), srcs{k,2}, 'seeds');
    bs = B.(srcs{k,2}); sd = double(B.seeds);
    for j = 1:numel(bs)
        m = find(want == sd(j));
        if ~isempty(m) && got(m(1)) == 0
            subset10base{m(1)} = bs{j};
            got(m(1)) = 1;
        end
    end
end
assert(all(got==1), '缺基座: %s', mat2str(want(got==0)));
for k = 1:numel(subset10base)
    b = subset10base{k};
    assert(size(b.candidate0,1)==200 && size(b.testset,1)==5000 && size(b.testsample,1)==12000 ...
        && size(b.init_x,1)==25, 'seed %d 基座尺寸异常', b.rs);
    fprintf('seed %3d: candidate0=200x10 testset=5000x10 testsample=12000x10 init=25x10\n', b.rs);
end
seeds = want;
save(fullfile(pwd,'subset10_ww_base.mat'), 'subset10base', 'seeds', '-v7');
fprintf('10 种子基座合并完成: subset10_ww_base.mat\n');
fprintf('=== make_wingweight_subset10_base DONE OK ===\n');
