%
% lines = my_cat(file_name)
%
% Simulate the function of "cat" in Unix systems. 
% The input is a file name.
% The output is a cell array, each cell contains one line of the file
% content. 
% Author: Xiong Xiao, 
%         School of Computer Engineering, Nanyang Technological University
% Date Created: 23 Oct, 2008
% Last Modified: 23 Oct, 2008
%
function lines = my_cat(file_name)

FID = fopen(file_name, 'r');

lines = read_line_by_line(FID);

fclose(FID);


