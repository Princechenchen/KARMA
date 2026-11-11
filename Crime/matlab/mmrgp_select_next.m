function selected_index = mmrgp_select_next(method,train_x,train_y,candidate,reference,model_id,step,seed,gpml_root)
% Dispatch one real-model acquisition; returned index is MATLAB one-based.
assert(~isempty(regexp(model_id,'^[A-Za-z0-9_]+$','once')),'Invalid model_id');
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
for folder={'numeric_sample','sampling_Funciton','MS-PSO','MS-PSO-pro','src','MRSM_subprograms1','MRSM_subprograms1_correxp'}
    addpath(fullfile(root,folder{1}));
end
% Original samplers opt in to the portable correxp model through this variable.
assignin('base','crimes_gpml_root',gpml_root);
name=['crimes_mmrgp_',model_id]; state_name=['crimes_state_',model_id];
assert(evalin('base',sprintf('exist(''%s'',''var'')',name))==1,'Model is not fitted');
if evalin('base',sprintf('exist(''%s'',''var'')',state_name))==1
    model=evalin('base',state_name);
else
    model=struct('init_x',train_x,'init_value',train_y,'n_init',size(train_x,1));
end
model.now_x=train_x; model.now_value=train_y; model.MMRGP=evalin('base',name);
model.ClassPos=[1 2]; model.dim=2; model.total_iter=100;
model.candidate=candidate; model.candidate0=candidate; model.testsample=reference;
model.gpml_root=gpml_root; model.c4_seed=seed+1009*step;
model.c10_seed=seed+1009*step; model.c10_loo_exact=false;
model.qbc_K=8; model.qbc_frac=0.8; model.hhs_failure_class=0;
rng(seed+1009*step,'twister');
K=evalin('base','crimes_knowledge_2021');
fn_hg=@(X) crimes_knowledge_eval(X,K,'pi_hg');
fn_bnd=@(X) crimes_knowledge_eval(X,K,'pi_bnd');
model.pi_comp_fns={fn_hg,fn_bnd};
model.pi_phys_fn=@(X) max(fn_hg(X),fn_bnd(X));
model.pi_comp_adaptive=false; model.pi_comp_eta=0.5; model.pi_comp_alpha_min=0.1;
model.pi_unif_grid=51; model.pi3_anneal=struct('k',5,'L',3,'tau_drop',0.25);
switch upper(method)
    case 'C2'
        [x_star,model]=mMeanMSE_norm(model);
    case 'C3'
        [x_star,model]=mIMSE_refset(model);
    case 'C4'
        [x_star,model]=PALROI_crimes(model);
    case 'C5'
        [x_star,model]=mAK_MCS_IRS_unified(model);
    case 'C6'
        [x_star,model]=mGlobal_SUR2w_varcs02lPD_fast_piBO(model);
    case 'C7'
        [x_star,model]=mROI_SUR2w_varcs02lPD_fast_piBO_DS(model);
    case 'C9'
        [x_star,model]=KeyRegion_sampling(model);
    case 'C10'
        [x_star,model]=Voronoi_MixedMetric(model);
    case 'C10-APPROX'
        [x_star,model]=Voronoi_MixedMetric(model);
    case 'C11'
        [x_star,model]=HHs_Mixed(model);
    case 'C12'
        [x_star,model]=mGlobal_SUR2w_varcs02lPD_fast_QBC(model);
    case 'ROI-D'
        roi_d_constant_knowledge=@(X) 0.5*ones(size(X,1),1); %#ok<NASGU>
        model.pi_comp_fns={roi_d_constant_knowledge,roi_d_constant_knowledge};
        model.pi_phys_fn=roi_d_constant_knowledge;
        model.pi_comp_alpha=[0 0]; % zero weights force pi3_active=false: pure data-ROI ablation
        model.pi3_active=false; model.pi_comp_adaptive=false;
        [x_star,model]=mROI_SUR2w_varcs02lPD_fast_PI_pil3br_DS_uconf(model);
    case 'KARMA'
        model.pi_comp_adaptive=true;
        [x_star,model]=mROI_SUR2w_varcs02lPD_fast_PI_pil3br_DS_uconf(model);
    case 'KARMA-FIXED'
        model.pi_comp_adaptive=false;
        model.pi_comp_alpha=[1 1];
        [x_star,model]=mROI_SUR2w_varcs02lPD_fast_PI_pil3br_DS_uconf(model);
    case 'KARMA-HG'
        model.pi_comp_active_mask=[1 0];
        model.pi_comp_adaptive=true;
        [x_star,model]=mROI_SUR2w_varcs02lPD_fast_PI_pil3br_DS_uconf(model);
    case 'KARMA-BND'
        model.pi_comp_active_mask=[0 1];
        model.pi_comp_adaptive=true;
        [x_star,model]=mROI_SUR2w_varcs02lPD_fast_PI_pil3br_DS_uconf(model);
    otherwise
        error('Unknown acquisition method: %s',method);
end
assert(isnumeric(x_star) && isequal(size(x_star),[1 2]) && all(isfinite(x_star)));
match=all(abs(candidate-x_star)<=1e-10,2);
assert(sum(match)==1,'Selected point must match exactly one unobserved candidate');
selected_index=find(match,1);
if isfield(model,'last_roi_table') && ~isempty(model.last_roi_table)
    occupied=table2array(model.last_roi_table); occupied=occupied(:,1:2*model.dim);
    in_any=false(size(candidate,1),1);
    for i=1:size(occupied,1)
        upb=occupied(i,2:2:end); dob=occupied(i,1:2:end);
        in_any=in_any | all(candidate<=upb & candidate>=dob,2);
    end
    if ~isfield(model,'roi_selection_hist') || isempty(model.roi_selection_hist)
        model.roi_selection_hist=zeros(0,4);
    end
    model.roi_selection_hist(end+1,:)=[step,double(in_any(selected_index)),sum(in_any),sum(~in_any)];
end
assignin('base',state_name,model);
end
