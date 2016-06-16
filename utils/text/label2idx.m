function idx = label2idx(vocab, label)

V = length(vocab);

idx = zeros(1,length(label));
for i=1:length(label)
    found = searchInCellBinary(vocab, label{i}, 1, V);
    if found
        idx(i) = found;
    else
        fprintf('Error: symbol %s not found in vocab!\n', label{i});
    end
end
end