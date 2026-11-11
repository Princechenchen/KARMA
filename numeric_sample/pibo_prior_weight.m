function weight = pibo_prior_weight(pi_values, n_obs, beta)
%PIBO_PRIOR_WEIGHT  Compute the piBO decaying prior multiplier.
%   weight(x) = pi(x)^(beta / n_obs).
%   n_obs is the one-based sequential iteration index after the initial
%   design D_0. The exponent is protected from n_obs=0.

    if nargin < 3 || isempty(beta)
        beta = 1;
    end
    if nargin < 2 || isempty(n_obs)
        n_obs = 1;
    end
    validateattributes(pi_values, {'numeric'}, {'real','finite','nonnegative'}, ...
        mfilename, 'pi_values');
    validateattributes(n_obs, {'numeric'}, {'real','finite','scalar','positive'}, ...
        mfilename, 'n_obs');
    validateattributes(beta, {'numeric'}, {'real','finite','scalar','nonnegative'}, ...
        mfilename, 'beta');

    n_obs = max(1, n_obs);
    pi_values = min(max(pi_values, eps), 1);
    weight = pi_values .^ (beta / n_obs);
end
