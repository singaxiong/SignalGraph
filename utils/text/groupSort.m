% This function sort the utterance ID, and also rearrange the corresponding
% data cell array. 
% 
function [ID_sort, data_sort] = groupSort(ID, data)

[ID_sort, idx] = sort(ID);
data_sort = data(idx);