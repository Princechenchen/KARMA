clear; close all;
root = pwd;
assert(endsWith(root,'wingweight'), '请在 wingweight/ 下运行');
addpath([root,'/../testFunc']);
rec_file = fullfile(root,'0422ww25050best.mat');
tvv = load(rec_file, 'modelROI_SURwvarclPD');
cells = tvv.modelROI_SURwvarclPD; clear tvv
warm = cells{1};
junk = MMRGP0(warm.init_value, warm.init_x, double(warm.ClassPos), double(warm.dim)); clear junk warm
for ci = [1 3 5]
    hj = cells{ci};
    X  = hj.now_x;  NV = hj.now_value;  ts = hj.testset;
    ClassPos = double(hj.ClassPos);
    [trainym, settings] = mapminmax(NV', -1, 1); trainym = trainym';
    ytrue = wingweight(ts);
    myc = mean(ytrue); myc(ClassPos==2) = 1;
    yfixed = ytrue ./ myc;
    e_arch = hj.error{20}{1};
    addpath(fullfile(root,'..','MRSM_subprograms_tmpcorr'));
    w1 = which('MRSM');
    m2 = MRSM(X, trainym); m2.settings = settings; m2.classpos = ClassPos;
    p2 = predict_resp(m2, ts);
    r2 = mean(abs(p2(:,1:2) - yfixed(:,1:2)), 1);
    addpath(fullfile(root,'..','MRSM_subprograms1'));
    w2 = which('MRSM');
    m1 = MRSM(X, trainym); m1.settings = settings; m1.classpos = ClassPos;
    p1 = predict_resp(m1, ts);
    r1 = mean(abs(p1(:,1:2) - yfixed(:,1:2)), 1);
    fprintf('cell %d | 存档RMAE=[%.6f %.6f]\n', ci, e_arch(1), e_arch(2));
    fprintf('  %-12s RMAE=[%.6f %.6f] | 与存档差 [%.2e %.2e]\n', w2, r1(2), r1(1), abs(r1(1)-e_arch(1)), abs(r1(2)-e_arch(2)));
    fprintf('  %-12s RMAE=[%.6f %.6f] | 与存档差 [%.2e %.2e]\n', w1, r2(2), r2(1), abs(r2(1)-e_arch(1)), abs(r2(2)-e_arch(2)));
end
fprintf('=== kernel_discriminate DONE ===\n');
