function model=cand_gen(model,ns,method)
nv=model.variables.dim;
    if strcmp('uniform', method)
        candidate = unifrnd(0,1,nv,ns);
    elseif strcmp('normal', method)
        candidate = normrnd(0.5,0.15,nv,ns);
    else
        error('Distribution types are not defined correctly!\n');
    end
candidate=candidate';
model.candidate=candidate;
