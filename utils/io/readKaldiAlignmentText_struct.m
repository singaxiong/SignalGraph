function [ali, vocab] = readKaldiAlignmentText_struct(file_name)
lines = my_cat(file_name);

vocab = [];
for i=1:length(lines)
    PrintProgress(i,length(lines),1000);
    idx = regexp(lines{i}, ' ');
    uttName = lines{i}(1:idx(1)-1);
    ali.(['U_' uttName]) = str2num(lines{i}(idx(1)+1:end))';
    vocab = [vocab; ali.(['U_' uttName])];
    if mod(i,100)==0
        vocab = unique(vocab);
    end
end
vocab = unique(vocab);
end
