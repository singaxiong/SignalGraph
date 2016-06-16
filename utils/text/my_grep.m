% 
% output = my_grep(input, pattern)
%
% Similate the function of grep in Unix
% The inputs are
%       input: texts in cell format
%       pattern: the pattern that is searched for
% The output is
%       output: lines in input that contains the pattern
%
% Author: Xiong Xiao, 
%         School of Computer Engineering, Nanyang Technological University
% Date Created: 23 Oct, 2008
% Last Modified: 23 Oct, 2008
%
function output = my_grep(input, pattern)

N = length(input);  

output{1} = '';
cnt = 1;
for i=1:N
    if length(regexp(input{i},pattern)) > 0
        output{cnt} = input{i};
        cnt = cnt + 1;
    end    
end