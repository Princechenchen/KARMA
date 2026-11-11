function r = hpo_eval_online(x_row, cell_id)
xstr = sprintf('%.12g,', x_row(:)');
xstr(end) = [];
lastErr = [];
for attempt = 1:3
    try
        s = char(py.hpo_eval.eval_online(py.str(xstr), py.str(sprintf('%d', cell_id))));
        r = jsondecode(s);
        assert(isfield(r, 'y1') && isfield(r, 'train_acc'), '桥返回字段缺失');
        return;
    catch err
        lastErr = err;
        fprintf('[hpo_eval_online] 第 %d 次失败: %s\n', attempt, err.message);
        pause(2);
    end
end
error('hpo_eval_online:Failed', '在线评估 3 次失败: %s | %s', ...
    lastErr.identifier, lastErr.message);
end
