function mmrgp_set_knowledge(x,y,pi_hg,pi_bnd)
% Store the 2021-only knowledge field for this MATLAB session.
x=x(:); y=y(:); pi_hg=pi_hg(:); pi_bnd=pi_bnd(:);
assert(numel(x)==numel(y) && numel(x)==numel(pi_hg) && numel(x)==numel(pi_bnd));
assert(all(isfinite([x;y;pi_hg;pi_bnd])));
K=struct('x',x,'y',y,'pi_hg',pi_hg,'pi_bnd',pi_bnd);
K.Fhg=scatteredInterpolant(x,y,pi_hg,'linear','nearest');
K.Fbnd=scatteredInterpolant(x,y,pi_bnd,'linear','nearest');
assignin('base','crimes_knowledge_2021',K);
end
