% Generate random blocks of sentences and also random frame index in each
% block.
function [sentIdxInBlock] = shuffle_data(layer, para, data, isDev)
if nargin<4;    isDev = 0; end
% Decide the block size
max_block_size = decide_max_block_size_tree2(layer, para);
nSent = length(data(1).data);
rng('shuffle');

frameInSent = 0;
for j=1:length(para.IO.inputFeature)
    if para.IO.inputFeature(j) == 0
        tmpFrameInSent{j} = para.IO.avgFramePerUtt;
    else
        tmpFrameInSent{j} = cellfun('size', data(j).data,2);
    end
    frameInSent = max(tmpFrameInSent{j}, frameInSent);
end

avgFramePerSent = sum(frameInSent)/nSent;
nSentInBlock = para.NET.maxNumSentInBlock;
if nSentInBlock*avgFramePerSent > max_block_size
    nSentInBlock = max(1,floor(max_block_size / avgFramePerSent));
end
nBlock = ceil(nSent/double(nSentInBlock));

% Assign sentences to blocks
if para.IO.ClassLabel4EvenBlock>0   % the blocks should contain as diverse classes as possible. To use this option, each sentence should have exactly one class label.
    label = cell2mat(data(para.IO.ClassLabel4EvenBlock).data);
    sentIdxInBlock = AssignSamples2BlocksByLabel(label, nBlock, nSentInBlock, para.IO.ClassLabel4EvenBlock_refill, isDev);
else
    if para.NET.randomizedBlock==0
        randomSentIdx = 1:nSent;
    else
        randomSentIdx = randperm(nSent);
    end
    actual_nSentInBlock = [];
    for i=1:nBlock
        actual_nSentInBlock(i) = min(nSentInBlock, nSent-sum(actual_nSentInBlock(1:i-1)));
        sentIdxInBlock{i} = randomSentIdx( sum(actual_nSentInBlock(1:i-1))+1 : sum(actual_nSentInBlock(1:i)) );
    end
end
end
