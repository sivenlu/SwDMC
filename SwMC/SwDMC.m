function [ClusteringResult, S, clusternum] = SwDMC(X,y0,c)
% INPUT:
% X: constructed affinity matrices, X is a cell and X{i} is n by n
% y0: cluster labels
% c: cluster number
% OUTPUT:
% ClusteringResult: ACC, NMI, Purity
% S: doubly stochastic similariy matrix.
% Ref:
% Feiping Nie, Jing Li, Xuelong Li.
% Self-weighted Multiview Clustering with Multiple Graphs.
% The 26th International Joint Conference on Artificial Intelligence, Melbourne, AUS, 2017.
[~,viewnum] = size(X);
alpha = rand(1,viewnum);
alpha = alpha/sum(alpha,2);
NITER = 50; 
eps = 1e-8;
lambda = 1; 
Obj = zeros(50, 1);
%%
for iter = 1 : NITER 
    % Fix alpha, update S. 
    if iter == 1
       [~, S] = CLR(alpha, X, c, lambda); 
    else
       [~, S] = CLR(alpha, X, c, lambda, S0); 
    end
    % Fix S, update alpha 
    for v = 1 : viewnum
        alpha(1, v) = 0.5 / norm(S - X{v}, 'fro');
    end
    S0 = S;
      
    % Calculate obj
    obj = 0;
    for v = 1 : viewnum
        obj = obj + norm(S - X{v}, 'fro');
    end
    Obj(iter) = obj;
    if iter > 1 && abs(Obj(iter-1) - Obj(iter)) < eps
        break;
    end   
end
S = (S + S') / 2;
S = Marcus_Mapping(S);
[clusternum, y]=graphconncomp(sparse(S)); y = y';
% if clusternum ~= c
%     sprintf('Can not find the correct cluster number: %d', c)
% end
ClusteringResult = ClusteringMeasure(y0, y); %ACC NMI PUR
Tag = isequal(Obj, sort(Obj, 'descend'));
