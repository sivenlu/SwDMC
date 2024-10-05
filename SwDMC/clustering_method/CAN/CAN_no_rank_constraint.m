% min_{A>=0, A*1=1, F'*F=I}  trace(D'*A) + r*||A||^2
% just relax the rank constraint of CAN algorithm
function [A] = CAN_no_rank_constraint(X, k, r, islocal)
% X: dim*num data matrix, each column is a data point
% k: number of neighbors to determine 
%    the initial graph and the parameter r(if r <= 0)
% r(gamma): paremeter, which could be set to a large enough value.
%    If r<0, then it is determined by algorithm with k
%    r(gamma) is related to the sparsity of S
% islocal: 
%           1: only update the similarities of the k neighbor pairs, faster
%           0: update all the similarities
% A: num*num learned symmetric similarity matrix

num = size(X, 2);
if nargin < 5
    islocal = 1;
end
if nargin < 4
    r = -1;
end
if nargin < 3
    k = 15;
end

distX = L2_distance_1(X, X);
[distX1, idx] = sort(distX,2);
rr = zeros(num, 1);

% if the input r <= 0, initial r by using the relationship between r and k
% For more details, see the Eq. (33) of the corresponding paper.
if r <= 0
    for i = 1:num
        di = distX1(i, 2:k+2);
        rr(i) = 0.5 * (k*di(k+1) - sum(di(1:k)));
    end
    r = mean(rr);
end

A = zeros(num);
for i=1:num
    if islocal == 1
        idxa0 = idx(i, 2:k+1);
    else
        idxa0 = 1:num;
    end
    dxi = distX(i, idxa0);
    ad = -dxi / (2 * r);
    A(i, idxa0) = EProjSimplex_new(ad);
end
% ensure A is symmetric
A = (A + A') / 2;


