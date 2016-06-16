%
% Perform the mean and variance normalization (MVN)
%       output = MVN(input,mean_d,var_d,weight)
%
% Input: 
%   input: a feature vector or matrix to be processed. If the input is a
%          vector, it should be a column vector; if the input is a matrix,
%          it should be MxN, where M is the number of frames and N is the
%          feature index.
%   mean_d: the desired mean of the output. If the mean_d is not provided,
%           zero is used. If the input is a vector, mean_d should be a
%           scalar; if the input is a matrix, mean_d can be either a scalar
%           or a column vector.
%   var_d: the desired variance of the output. If the var_d is not provided,
%          one is used. If the input is a vector, var_d should be a
%          scalar; if the input is a matrix, var_d can be either a scalar 
%          or a column vector.
%   weight: the weight of each frame during the normalization. The
%          function supports partial ownership, i.e. a frame may be owned by a
%          class partially. This is useful in multi-class MVN. If weight is not
%          provided, all frames will have weight 1, i.e. conventional MVN.
%
% Output:
%   output: the mean-variance-normalized features
%
% Author: Xiao Xiong, School of Computer Engineering, NTU, Singapore.
% Created: Jan, 2005
% Last Modified: 21 Dec, 2010
%
function [output] = MVN(input,mean_d,var_d, weight)
useWeight = 1;
if nargin < 4
    useWeight = 0;
end
if nargin < 3
    var_d = 1;  % default variance is 1
end
if nargin < 2
    mean_d = 0; % default mean is 0
end
[M, N] = size(input);   % Perform MVN for each column individually

if useWeight
    [meanX,varX] = findMeanVarainceWeighted(input, weight, 0);      % find the mean and variance of input weighted by their weights
    output = input - repmat(meanX',M,1);
    gain = diag(sqrt(var_d./varX));
    output = output * gain;
    output = output + repmat(mean_d',M,1); 
else
    output = CVN(CMN(input,mean_d),var_d);
end
