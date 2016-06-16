%
% write_matrix(FID,data)
%
% This function write a matrix to the specified file. Both two dimensional
% matrix and one dimensional vector are accepted. 
%
% Author: Xiao Xiong
% Created: 6 Feb, 2007
% Last modified: 6 Feb, 2007

function write_matrix_file(filename, data)

FID = fopen(filename, 'w');
write_matrix(FID, data);
fclose(FID);

