function notBPable = IsNotBackPropagatable(node)

updatable = {'input', 'idx2vec', 'enframe', 'comp_gcc', 'stft'};

notBPable = 0;
for i=1:length(updatable)
    if strcmpi( node, updatable{i} )
        notBPable = 1;
        break;
    end
end

end
