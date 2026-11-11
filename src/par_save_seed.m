function par_save_seed(fseed, S)
fn = fieldnames(S);
save(fseed, fn{:}, '-v7');
end
