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
function [output]= CMN_online(input,mean_d)
if nargin==1
    mean_d=0;
end

[M,N] = size(input);

% track the mean
window_size = 201;
local_mean = MovingAverageFilter(input, window_size);

% output = input - repmat(mean(input)-mean_d',M,1);
output = input - local_mean;

end