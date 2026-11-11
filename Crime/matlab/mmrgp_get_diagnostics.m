function diagnostics = mmrgp_get_diagnostics(model_id)
% Return compact acquisition traces without exporting large ROI/model objects.
assert(~isempty(regexp(model_id,'^[A-Za-z0-9_]+$','once')),'Invalid model_id');
diagnostics=struct();
state_name=['crimes_state_',model_id];
if evalin('base',sprintf('exist(''%s'',''var'')',state_name))~=1
    return
end
model=evalin('base',state_name);
fields={'pi_comp_alpha','pi_comp_cons_hist','pi_comp_alpha_hist','pi3_hist', ...
    'roi_selection_hist','qbc_hist','hhs_beta1_history','hhs_beta2_history','hhs_CP_history'};
for i=1:numel(fields)
    if isfield(model,fields{i})
        diagnostics.(fields{i})=model.(fields{i});
    end
end
if isfield(model,'C10')
    diagnostics.C10=struct('loo_exact',model.C10.loo_exact,'loo_cap',model.C10.loo_cap, ...
        'entropy_weights',model.C10.entropy_weights,'selected_cell',model.C10.selected_cell);
end
end
