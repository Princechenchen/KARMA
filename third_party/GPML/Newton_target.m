% Evaluate criterion Psi(alpha) = alpha'*K*alpha + likfun(f), where 
% f = K*alpha+m, and likfun(f) = feval(lik{:},hyp.lik,y,  f,  [],inf).

% targetfunction = logp(y|f)-1/2*f'inv(K)f;   求这个函数
function [psi,dpsi,f,alpha,dlp,W] = Newton_target(alpha,m,K,y)
  f = K*alpha+m; yf = y.*f;                            % product latents and labels
  [lp,dlp,d2lp] = logphi(yf); dlp = y.*dlp; W = -d2lp;
  psi = alpha'*(f-m)/2 - sum(lp);
  if nargout>1, dpsi = K*(alpha-dlp); end
end