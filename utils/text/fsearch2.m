% seek for a specified character and stop
% Author: Xiao Xiong
% Created: 10 Jul 2006
% Last modified: 10 Jul 2006

function endOfFile = fsearch2(target, fid)
target_length = length(target);
endOfFile = 0;
while 1
    found_target = 1;
    for i=1:target_length
        tline = fgets(fid, 1);
        if ~ischar(tline)
            endOfFile = 1;
            return;
        end
        if strcmp(tline,target(i)) == 0
            found_target = 0;
            break;
        end
    end
    if found_target
        endOfFile = 0;
        return;
    else
        fseek(fid,1-i,'cof');
    end            
%     disp(tline);
end