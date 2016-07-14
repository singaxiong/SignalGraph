% We would like to sort the training segments based on their duration. The
% purpose is to make the sentences in a minibatch to have similar lengths
%
function sentIdx = AssignSamples2BlocksByDuration(data, nBlock, nSampleInBlock)

nFr = cellfun('size', data,2);

[nFrs,sortedIdx] = sort(nFr);

for i=1:nBlock
    idx1 = (i-1)*nSampleInBlock+1;
    idx2 = min(length(data), i*nSampleInBlock); 
    sentIdx{i} = sortedIdx(idx1:idx2);
end

% we should shuffle the block ids such that we don't always start with
% short sentences and then long sentences. 
randIdx = randperm(nBlock);
sentIdx = sentIdx(randIdx);

end

