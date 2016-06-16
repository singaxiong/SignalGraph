%
% filtered_list = filter_uttr_list(utterance_list,method, step/total)
%
% Create a subset of the input list by evenly sample the entries from the
% input list. There are two ways to do the task:
%   1. Specify the gap between the sampled entries. The gap will be "step-1".
% The settings should be: 
%       method = 'even'; step = gap+1;
%   2. Specify the number of entries in the output list. In this case, the
% gap may not be an integer and the output list is not truely evenly
% sampled fromthe input list. The settings should be:
%       method = 'total'; step = number of entries in output list
%
% Author: Xiao Xiong
% Created: 7 Feb, 2007
% Last Modified: 7 Feb, 2007

function filtered_list = filter_uttr_list(utterance_list,method,step)

len_utt = length(utterance_list);
if len_utt == 0
    filtered_list = [];
    return;
end
switch method
    case 'even'
        for i=1:step:len_utt
            cnt = (i-1)/step+1;
            if iscell(utterance_list)
                filtered_list{cnt} = [utterance_list{i}];
            else
                filtered_list(cnt) = utterance_list(i);
            end
        end
    case {'total', 'Total'}
        N_desired_enty = step;
        temp = len_utt/N_desired_enty;
        for i=temp:temp:len_utt
            cnt = round(i/temp);
            i = round(i);
            if iscell(utterance_list)
                filtered_list{cnt} = [utterance_list{i}];
            else
                filtered_list(cnt) = utterance_list(i);
            end
        end
end



