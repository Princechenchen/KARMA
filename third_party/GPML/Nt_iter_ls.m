% Run IRLS Newton algorithm to optimise Psi(alpha).
function [alpha,L,Psi_new,ldB2,lp,dlp,W ]= Nt_iter_ls(alpha, m,K,y)
% max no of Newton steps
maxit = 20;                       % default value
% min likelihood curvature
Wmin = 0.0;                                       % default value
% stop Newton iterations
tol = 1e-6;                                       % default value

  smin_line = 0; smax_line = 2;           % min/max line search steps size range
  nmax_line = 10;                          % maximum number of line search steps
  thr_line = 1e-4;                                       % line search threshold
  Psi_line = @(s,alpha,dalpha) Newton_target(alpha+s*dalpha,m,K,y);% line search
  pars_line = {smin_line,smax_line,nmax_line,thr_line};  % line seach parameters
  search_line = @(alpha,dalpha) brentmin(pars_line{:},Psi_line,5,alpha,dalpha);
  
  f = K*alpha+m;  yf = y.*f;                            % product latents and labels
  [lp,dlp,d2lp] = logphi(yf); W = -d2lp; n = size(K,1); dlp = dlp.*y;
  Psi_new = Newton_target(alpha,m,K,y);
  Psi_old = Inf;  % make sure while loop starts by the largest old objective val
  it = 0;                          % this happens for the Student's t likelihood
  while Psi_old - Psi_new > tol && it<maxit                       % begin Newton
    Psi_old = Psi_new; it = it+1;
    % limit stepsize
    W = max(W,Wmin); % reduce step size by increasing curvature of problematic W
    % symmetric B = I+sW*K*sW
    sW = sqrt(W); 
    L = chol(eye(n)+sW*sW'.*K,"lower");             % Cholesky factor of B
%     L = chol(eye(n)+sW*sW'.*K);             % Cholesky factor of B
    ldB2 = sum(log(diag(L)));                                    % log(det(B))/2

    b = W.*(f-m) + dlp;
%     alpha = L'\(L\y);
    dalpha = b - L'\(L\(K*b.*sW)).*sW - alpha;    % Newton direction + line search
    [s_line,Psi_new,n_line,dPsi_new,f,alpha,dlp,W] = search_line(alpha,dalpha);
  end      % end Newton's iterations 
end