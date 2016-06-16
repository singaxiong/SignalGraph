% seperate a big matrix of featuers into individual utterances. The length
% of utterances are defined in nFr. 
%
function [utt] = FeatureBlockToUttCell(block, nFr)

[nFrTotal,dim] = size(block);

if nFrTotal ~= sum(nFr)
    fprintf('Error: total number of frames does not agree!\n');
    return;
end

offset = 0;
for i=1:length(nFr)
    utt{i} = block(offset+1:offset+nFr(i),:);
    offset = offset + nFr(i);
end
