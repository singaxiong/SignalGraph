%
% data = read_matrix(FID,M,N)
%
% This function read a two dimensional matrix from the file specified. The
% dimension of the matrix need to be specified. The one dimensional vector
% can be accepted as a row matrix or column matrix.
%
% Author: Xiao Xiong
% Created: 6 Feb, 2007
% Last modified: 6 Feb, 2007

function data = read_matrix(FID,M,N)
% M and N are the number of rows and columns respectively
data = zeros(M,N);
for i=1:M
    line = fgetl(FID);
    temp = textscan(line, '%n', N);
    data(i,:) = temp{1};
end