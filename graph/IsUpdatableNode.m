function isUpdatable = IsUpdatableNode(node)

updatable = {'Affine', 'weighting', 'LSTM', 'word2vec', 'tconv'};

isUpdatable = 0;
for i=1:length(updatable)
    if strcmpi( node, updatable{i} )
        isUpdatable = 1;
        break;
    end
end

end
