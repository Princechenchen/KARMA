files = {'make_hpo_base.m', 'hpo_eval_online.m', 'pi_hpo_prior.m', ...
         'mGlobal_SUR2w_varcs02lPD_fast_EI.m', 'main_compare_hpo_mixed.m', ...
         'smoke_hpo_driver.m'};
tot = 0;
for k = 1:numel(files)
    r = checkcode(files{k}, '-severity');
    n = numel(r);
    tot = tot + n;
    fprintf('%-42s : %d 条\n', files{k}, n);
    for j = 1:min(n, 5)
        fprintf('    L%-4d [sev%d] %s\n', r(j).line, r(j).severity, r(j).message);
    end
end
fprintf('LINT TOTAL %d 条(非零多为风格项; 重点看有无语法/解析错误)\n', tot);
disp('LINT ALLDONE');
