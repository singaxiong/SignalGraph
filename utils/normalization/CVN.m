%
% Perform the cepstral variance normalization (CVN)
%       output = CVN(input,var_d)
%
% Input: 
%   input: a feature vector or matrix to be processed. If the input is a
%          vector, it should be a column vector; if the input is a matrix,
%          it should be MxN, where M is the number of frames and N is the
%          feature index.
%   var_d: the desired variance of the output. If the var_d is not provided,
%          one is used. If the input is a vector, var_d should be a
%          scalar; if the input is a matrix, var_d can be either a scalar 
%          or a column vector.
%
% Output:
%   output: the variance-normalized feature
% Author: Xiao Xiong, School of Computer Engineering, Nanyang Tech. Univ.
% Created: Jan, 2005
% Last Modified: 19 Oct, 2007
%
function output = CVN(input,var_d,weight)
if nargin==1
    var_d=1;
end

[M,N] = size(input);

if nargin == 3
    [meanX,varX] = findMeanVarainceWeighted(input, weight, 0);      % find the mean and variance of input weighted by their weights
else
    meanX = mean(input);
    varX = var(input);
end

if length(var_d)==1
    output = (input-repmat(meanX,M,1)) ./ repmat(   sqrt(varX/var_d),   M,1);
else
    output = (input-repmat(meanX,M,1)) ./ repmat(   sqrt(varX./var_d'),   M,1);
end
output = output+repmat(meanX,M,1);
