% gp_check_gradients  自检脚本:NLL 解析梯度 vs 中心差分
% 运行(在本目录下): matlab -batch gp_check_gradients
% 判据:每个核的最大相对误差 < 1e-5,否则报错退出(便于 -batch 的 rc 判定)
rng(0);
n = 12;  d = 2;
X = rand(n,d);  y = randn(n,1);
covlist = {'gpCovSEard','gpCovSEiso','gpCovMatern52'};
allok = true;
for c = 1:numel(covlist)
    covname = covlist{c};
    if strcmp(covname,'gpCovSEiso'), ncov = 2; else, ncov = d+1; end
    th  = 0.3*randn(ncov+2,1);                     % [cov; mean; lik]
    nlZ = @(t) gp_nll(mkhyp(t,ncov), str2func(covname), 'const', X, y);
    [f0, g] = nlZ(th);                              %#ok<NASGU>
    h  = 1e-5;  fd = zeros(numel(th),1);
    for ii = 1:numel(th)
        tp = th;  tp(ii) = tp(ii) + h;
        tm = th;  tm(ii) = tm(ii) - h;
        fd(ii) = (nlZ(tp) - nlZ(tm))/(2*h);
    end
    err = max(abs(fd - g)./max(1, abs(fd)));
    if err < 1e-5, tag = 'PASS'; else, tag = 'FAIL'; allok = false; end
    fprintf('%-16s  max rel err = %.3e   %s\n', covname, err, tag);
    fprintf('   analytic: %s\n', num2str(g.','%.4f '));
    fprintf('   finit-df: %s\n', num2str(fd.','%.4f '));
end
if ~allok, error('gp_check_gradients: FAILED'); end
fprintf('ALL GRADIENT CHECKS PASS\n');

function hyp = mkhyp(th, ncov)
hyp = struct('cov',th(1:ncov),'mean',th(ncov+1),'lik',th(ncov+2));
end
