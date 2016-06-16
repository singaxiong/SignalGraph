function [uttName_sorted, aliID_sorted, vocab] = readKaldiAlignmentText(file_name)
lines = my_cat(file_name);

vocab = [];
for i=1:length(lines)
    PrintProgress(i,length(lines),1000);
    idx = regexp(lines{i}, ' ');
    uttName{i} = lines{i}(1:idx(1)-1);
    aliID{i} = str2num(lines{i}(idx(1)+1:end))';
    vocab = [vocab; aliID{i}];
    if mod(i,100)==0
        vocab = unique(vocab);
    end
end
vocab = unique(vocab);

[uttName_sorted, order] = sort(uttName);
aliID_sorted = aliID(order);
