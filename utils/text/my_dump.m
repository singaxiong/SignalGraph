% 
% my_dump(file, lines)
%
% Similate the function of "right arrow file writing" function  in Unix
% The inputs are
%       file: the destination
%       lines: the text that will be written
%
% Author: Xiong Xiao, 
%         School of Computer Engineering, Nanyang Technological University
% Date Created: 23 Oct, 2008
% Last Modified: 23 Oct, 2008
%
function my_dump(file, lines, start, stop)
if nargin < 4
    stop = length(lines);
elseif stop > length(lines)
    fprintf('\tError: the ending pointer is larger than the size of the text\n\n');
end
if nargin < 3
    start = 1;
end

FID = fopen(file, 'w');

for i=start:stop
    if length(lines{i}) == 0
        fprintf(FID,'\n');
    elseif strcmp(lines{i}(end), '\n') == 1
        fprintf(FID, '%s', lines{i});
    else
        fprintf(FID, '%s\n', lines{i});
    end
end

fclose(FID);