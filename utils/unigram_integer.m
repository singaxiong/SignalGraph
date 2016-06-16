function [unigram, vocab] = unigram_integer(data, vocab)
if exist('vocab')==0 || length(vocab)==0
    vocab = unique(data);
end

for i=1:length(vocab)
    unigram(i) = sum(data==vocab(i));
end

end