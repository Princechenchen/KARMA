function values = crimes_knowledge_eval(query_x,K,channel)
% Linear interpolation with nearest-value extrapolation outside 2021 support.
assert(size(query_x,2)==2,'query_x must be N-by-2');
switch channel
    case 'pi_hg'
        values=K.Fhg(query_x);
    case 'pi_bnd'
        values=K.Fbnd(query_x);
    otherwise
        error('Unknown knowledge channel: %s',channel);
end
values=min(max(values(:),0.01),1);
end
