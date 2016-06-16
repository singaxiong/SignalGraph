% This function select utterance names from ID cell array that also appears
% in key cell array. It also select the corresponding features in data
% arrya. 
% The function assumes key is already sorted.
% [ID_selected, data_selected] = selectData(key, ID, data, FollowKeyIdx)
%
function [ID_selected, data_selected] = selectData(key, ID, data, FollowKeyIdx)
if nargin<4
    FollowKeyIdx = 0;
end

if FollowKeyIdx==0
    ID_selected = {};
    data_selected = {};
end
for i=1:length(ID)
    found = searchInCellBinary(key, ID{i}, 1, length(key));
    if found > 0
        if FollowKeyIdx
            ID_selected{found} = ID{i};
            data_selected{found} = data{i};
        else
            ID_selected{end+1} = ID{i};
            data_selected{end+1} = data{i};
        end
    end
end