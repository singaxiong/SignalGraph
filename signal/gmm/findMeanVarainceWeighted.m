%
% Find the mean and variance of weighted input data 
%       [m,v] = findMeanVarainceWeighted(X, weight, fullCov)
%
% Input: 
%   X: a feature vector or matrix to be processed. If the input is a
%          vector, it should be a column vector; if the input is a matrix,
%          it should be MxN, where M is the number of frames and N is the
%          feature index.
%   weight: the weight of each frame during the calculation. The
%          function supports partial ownership, i.e. a frame may be owned by a
%          class partially.
%   fullCov: set to 1 if full covariance matrix is needed.
%
% Output:
%   m: mean
%   v: variance or covariance matrix
%
% Author: Xiao Xiong, School of Computer Engineering, NTU, Singapore.
% Created: Sep, 2010
% Last Modified: 22 Dec, 2010
%
function [m,v] = findMeanVarainceWeighted(X, weight, fullCov)

if nargin <3
    fullCov =1;
end

[nFr, dim] = size(X);

if 1    % do pruning
    idx = find(weight>1e-3);
    weightOld = weight;
    weight = weight(idx);
    X = X(idx,:);
end

weight = weight(:);
weightSum = sum(weight);

if weightSum == nFr   % i.e., the weight = 1 for all frames, reduced to normal ML estimation
    m = mean(X)';
    if fullCov==1
        v = cov(X);
    else
        v = var(X)';
    end
else
    % find mean
    m = weight' * X / weightSum;
    m = m';

    % find variance
    X_norm = bsxfun(@minus, X, m');
    X_norm_weighted = bsxfun(@times, X_norm, sqrt(weight));
    
    v = X_norm_weighted' * X_norm_weighted / weightSum;
%         v = sum(X_norm_weighted.*X_norm_weighted)' / weightSum;
    if fullCov==0
        v = diag(v);
    end
end
    
end
