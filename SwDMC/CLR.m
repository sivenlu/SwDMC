function [y, S] = CLR(alpha, X, c,lambda,S0)
% process of Largrange solution
% INPUT:
% alpha: A vector containing the weights for each view.
% S0: Similarity matrix to be improved.
% c: Cluster number.
% OUTPUT:
% y: Predicted labels.
% S: The refined similarity matrix.
% Ref: 
%   Equation (16) of Self-weighted Multiview Clustering with Multiple Graphs  
viewnum = size(alpha, 2);
datanum = size(X{1}, 1);
if nargin < 5
   S0 = zeros(datanum);
   for v = 1 : viewnum
       S0 = S0 + alpha(1, v) * X{v};
   end
end
% get laplacian matrix.
S0 = S0 - diag(diag(S0));
S10 = (S0 + S0') / 2;
D10 = diag(sum(S10));
L0 = D10 - S10;

NITER = 50;
eps = 10e-11;

% get eigenvectors F corresponding to the c-smallest eigenvalues.
[F0, ~, ~] = eig1(L0, datanum, 0);
F = F0(:, 1:c);

for iter = 1 : NITER
    dist = L2_distance_1(F', F');
    S = zeros(datanum);
    % for each data point
    for i = 1 : datanum
        a0 = zeros(1, datanum);
        % for each view
        for v = 1 : viewnum
            temp = X{v};
            a0 = a0 + alpha(1, v) * temp(i, :);
        end    
        idxa0 = find(a0 > 0);
        ai = a0(idxa0);
        di = dist(i, idxa0);
        ad = (ai - 0.5 * lambda * di) / sum(alpha);
        S(i, idxa0) = EProjSimplex_new(ad);
    end
     
    A = S;
    A = A - diag(diag(A));
    A = (A + A') / 2;
    D = diag(sum(A));
    L = D - A;
    F_old = F; % store F temporally
    [F,~,ev] = eig1(L, c, 0);
    
    % check rank constraint.
    fn1 = sum(ev(1:c));
    fn2 = sum(ev(1:c+1));
    if fn1 > eps
        lambda = 2 * lambda;
    elseif fn2 < eps
        lambda = lambda / 2;
         F = F_old;
    else
        break;
    end    
end
 
[clusternum, y]=graphconncomp(sparse(A)); y = y';
if clusternum ~= c
    sprintf('Can not find the correct cluster number: %d', c)
end


