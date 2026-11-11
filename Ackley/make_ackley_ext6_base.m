clear; close all;
[parent_dir,~,~] = fileparts(pwd);
addpath([parent_dir,'/testFunc']);
seeds = [5 17 23 43 62 78];
nB = numel(seeds);
base6 = struct('rs',{},'candidate0',{},'init_x',{},'init_value',{},'testset',{},'testsample',{});
for b = 1:nB
    rs = seeds(b);
    rng(rs);
    candidate0 = rand(200,6);
    init_x     = lhsdesign(20,6);
    testset    = lhsdesign(5000,6);
    testsample = lhsdesign(12000,6);
    init_value = ackley(init_x);
    assert(all(size(candidate0)==[200,6]) && all(size(init_x)==[20,6]) ...
        && all(size(testset)==[5000,6]) && all(size(testsample)==[12000,6]) ...
        && all(size(init_value)==[20,3]), '基座尺寸异常');
    assert(all(init_x(:)>=0 & init_x(:)<=1) && all(candidate0(:)>=0 & candidate0(:)<=1) ...
        && all(testsample(:)>=0 & testsample(:)<=1), '基座越界');
    base6(b) = struct('rs',rs,'candidate0',candidate0,'init_x',init_x, ...
        'init_value',init_value,'testset',testset,'testsample',testsample);
    fprintf('base %d/6 (rs=%d): init 类均值=%.3f  y1=[%.3f %.3f]  y2=[%.3f %.3f]\n', ...
        b, rs, mean(init_value(:,3)), min(init_value(:,1)), max(init_value(:,1)), ...
        min(init_value(:,2)), max(init_value(:,2)));
end
recipe = ['rng(rs); candidate0=rand(200,6); init_x=lhsdesign(20,6); ' ...
    'testset=lhsdesign(5000,6); testsample=lhsdesign(12000,6); init_value=ackley(init_x). ' ...
    '配方按 rec0407Ackley_best.mat 结构指纹(2026-10-04); seeds=ship/Branin ext6 同款表 ' ...
    '(预承诺 [5 17 23 43 62 78]); cell=11..16'];
save(fullfile(pwd,'rec_ackley_ext6_base.mat'),'base6','seeds','recipe','-v7');
fprintf('已保存 rec_ackley_ext6_base.mat\n');
