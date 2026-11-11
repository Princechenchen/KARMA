clear; close all;
[parent_dir,~,~] = fileparts(pwd);
addpath([parent_dir,'/testFunc']);
seeds = [62 78];
nB = numel(seeds);
base4 = struct('rs',{},'candidate0',{},'init_x',{},'init_value',{},'testset',{},'testsample',{});
for b = 1:nB
    rs = seeds(b);
    rng(rs);
    candidate0 = rand(100,2);
    init_x     = lhsdesign(30,2);
    testset    = lhsdesign(5000,2);
    testsample = lhsdesign(12000,2);
    init_value = Branin_G(init_x);
    assert(all(size(candidate0)==[100,2]) && all(size(init_x)==[30,2]) ...
        && all(size(testset)==[5000,2]) && all(size(testsample)==[12000,2]) ...
        && all(size(init_value)==[30,3]), '基座尺寸异常');
    assert(all(init_x(:)>=0 & init_x(:)<=1) && all(candidate0(:)>=0 & candidate0(:)<=1) ...
        && all(testsample(:)>=0 & testsample(:)<=1), '基座越界');
    base4(b) = struct('rs',rs,'candidate0',candidate0,'init_x',init_x, ...
        'init_value',init_value,'testset',testset,'testsample',testsample);
    fprintf('base %d/2 (rs=%d): init 类别均值=%.3f  y1=[%.3f %.3f]  y2=[%.3f %.3f]\n', ...
        b, rs, mean(init_value(:,1)), min(init_value(:,2)), max(init_value(:,2)), ...
        min(init_value(:,3)), max(init_value(:,3)));
end
recipe = ['rng(rs); candidate0=rand(100,2); init_x=lhsdesign(30,2); ' ...
    'testset=lhsdesign(5000,2); testsample=lhsdesign(12000,2); init_value=Branin_G(init_x). ' ...
    '配方按 rec0407Branin_Gbest.mat 结构指纹(与 make_branin_ext4_base.m 一致, 2026-10-03); ' ...
    'seeds=ship ext6 表剩余2(预承诺 [62 78]); cell=15..16'];
save(fullfile(pwd,'rec_branin_ext2_base.mat'),'base4','seeds','recipe','-v7');
fprintf('已保存 rec_branin_ext2_base.mat\n');
