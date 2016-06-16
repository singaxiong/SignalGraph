%
% Perform the cepstral mean normalization (CMN)
%       output = CMN(input,mean_d)
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
%
% Output:
%   output: the mean-normalized feature
% Author: Xiao Xiong, School of Computer Engineering, Nanyang Tech. Univ.
% Created: Jan, 2005
% Last Modified: 19 Oct, 2007
%
function [output min_val max_val] = MinMaxNorm(input,min_val, max_val)
if nargin==1
    mean_d=0;
    min_val = min(input);
    max_val = max(input);
end

[M,N] = size(input);

output = input - repmat(min_val, M,1);
output = output ./ repmat(max_val - min_val, M, 1);