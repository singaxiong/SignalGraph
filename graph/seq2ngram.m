function output = seq2ngram(input, ngram_index, vocab)

% first, convert the input sequence to an index sequence starting from 1

for i = 1:length(vocab)
    input2(input==vocab(i)) = i;
end

vocab_size = length(vocab);

output = [];
for i=1:length(ngram_index)
    if ngram_index(i) == 1
        curr_ngram = zeros(vocab_size,1);
        for j=1:vocab_size
            curr_ngram(j) = sum(input2==vocab(j));
        end
        curr_ngram = curr_ngram / length(input);
    elseif ngram_index(i) == 2
        if 0
            curr_ngram2 = zeros(vocab_size);
            for j=2:length(input2)
                curr_ngram2(input2(j),input2(j-1)) = curr_ngram2(input2(j),input2(j-1)) +1;
            end
            curr_ngram2 = mat2vec(curr_ngram2);
        else
            curr_ngram = zeros(vocab_size^2,1);
            for j=2:length(input2)
                curr_idx = (input2(j-1)-1)*vocab_size + input2(j);
                curr_ngram(curr_idx) = curr_ngram(curr_idx) +1;
            end
        end
        curr_ngram = curr_ngram / max(1,(length(input)-1));
    elseif ngram_index(i) == 3
        if 0
            curr_ngram2 = zeros(vocab_size,vocab_size,vocab_size);
            for j=3:length(input2)
                curr_ngram2(input2(j),input2(j-1),input2(j-2)) = curr_ngram2(input2(j),input2(j-1),input2(j-2)) +1;
            end
            curr_ngram2 = reshape(curr_ngram2,vocab_size^3,1);
        else
            curr_ngram = zeros(vocab_size^3,1);
            for j=3:length(input2)
                curr_idx = (input2(j-2)-1)*vocab_size^2 + (input2(j-1)-1)*vocab_size + input2(j);
                curr_ngram(curr_idx) = curr_ngram(curr_idx) +1;
            end
        end
        curr_ngram = curr_ngram / max(1,(length(input)-2));
    end
    output = [output; curr_ngram];
end

output = sparse(output);

end