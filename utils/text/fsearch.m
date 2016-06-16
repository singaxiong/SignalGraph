% seek for a specified character and stop
% Author: Xiao Xiong
% Created: 22 Jan 2005
% Last modified: 22 Jan 2005

function endOfFile = fsearch(target, fid)
% fid = fopen('train.scp');
target_length = length(target);
endOfFile = 0;
while 1
    tline = fgets(fid, target_length);
    if ~ischar(tline)
        endOfFile = 1;
        break; 
    end
    if length(tline) == target_length
        if tline == target
            endOfFile = 0;
            break;
        end
    end
    %disp(tline);
end