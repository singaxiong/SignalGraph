% 
% my_append(file, lines)
%
% Similate the function of "double right arrow file writing" function in Unix
% The inputs are
%       file: the destination
%       lines: the text that will be written
%
% Author: Xiong Xiao, 
%         School of Computer Engineering, Nanyang Technological University
% Date Created: 02 Feb, 2009
% Last Modified: 02 Feb, 2009
%
function my_append(file, lines, start, stop)

FID = fopen(file, 'a');
if nargin < 3
    start = 1;
    stop = length(lines);
end

if iscell(lines)
    for i=start:min(stop,length(lines))
        if length(lines{i})==0
            continue;
        elseif strcmp(lines{i}(end), '\n') == 1
            fprintf(FID, '%s', lines{i});
        else
            fprintf(FID, '%s\n', lines{i});
        end
    end
else
    fprintf(FID, '%s\n', lines);
end

fclose(FID);

