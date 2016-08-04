% This function prepare the data for a block, which consists of many
% minibatches for DNN training. It can also be used to prepare data for
% cross-validation where each sentence may be a minibatch.
% Inputs:
%   visible: a cell array, each element is a feature file name or the
%            feature matrix.
%   target: similar to visible, but contain target.
%   aux_data: similar to visible, but contain auxiliary information for computing
%          visibule or cost.
%   para: a structure containing various configurations of the DNN
% Outputs:
%   block_visible: the prepared visible input for the DNN for current
%                  block.
%   block_target
%   block_scale
%
% Author: Xiong Xiao, NTU
% Date Created: 10 Oct 2013
% Last Modified: 24 Feb 2014
%
function [minibatch] = MinibatchPackaging_tree4(data, para)
if para.singlePrecision==0;    datatype = 'double';
else     datatype = 'single'; end

nStream = length(data);         nUtt = length(data(1).data);
feat = cell(nStream, nUtt);
for i = 1:nStream   % for each input stream
    for utt_i = 1:nUtt
        if para.IO.inputFeature(i)
            tmp_feat = data(i).data{utt_i};
        else    % if input is file names, read the actual features
            PrintProgress(utt_i, nUtt, max(50,ceil(nUtt/10)), sprintf('MinibatchPackaging_tree4: reading feature stream %d', i));
            tmp_feat = InputReader(data(i).data{utt_i},para.IO.fileReader(i), 0, para.useGPU, datatype);
        end
        if para.IO.isIndex(i)   % if the input is an index, retrieve the real data from the base.
            tmp_feat_idx = tmp_feat;
            switch para.IO.baseType
                case 'matrix'
                    tmp_feat = data(i).base(:,tmp_feat_idx);
                case 'tensor'
                    tmp_feat = data(i).base(:,:,tmp_feat_idx);
                case 'cell'
                    tmp_feat = data(i).base(tmp_feat_idx);
            end
        end
        if isfield(para, 'preprocessing') && length(para.preprocessing)>=i
            tmp_feat = FeaturePipe(tmp_feat, para.preprocessing{i});   % apply preprocessing
        end
        if strcmpi(datatype, 'single')
            tmp_feat = single(tmp_feat);
        else
            tmp_feat = double(tmp_feat);
        end
        feat{i,utt_i} = full(tmp_feat);
    end
end

% if length(feat)>1 && size(feat{2,1},1)==1 && para.DEBUG
%     fprintf('There are %d unique classes in this block\n', length(unique(cell2mat(feat(2,:)))));
% end
[feat, isVAD] = SynchronizeDataStreams3(feat, para);

if sum(para.IO.isTensor)>0      % At least one stream contains tensors, i.e. a training example is a matrix, e.g. a 2D image
    nSeqInMB = para.NET.nSequencePerMinibatch;
    nSeqInStreams = cellfun('size', feat,3);
    nSeqInStreams = sum(nSeqInStreams,2);
    nSeqInStreams(isVAD==1) = [];
    nSeq = nSeqInStreams(1);
    if sum(abs(nSeqInStreams - nSeqInStreams(1))); fprintf('Error: streams have different number of training samples\n'); end
    minibatch.nBatch = ceil(nSeq/nSeqInMB);
    if para.NET.variableLengthMinibatch
        minibatch.data = PackMinibatchTensorVariableLength(feat, nSeqInMB, nSeq, isVAD);
    else
        randSegIdx = randperm(nSeq);
        for i=1:nStream
            if isVAD(i); continue; end
            featInStream = cell2mat_tensor3D(feat(i,:));
            featInStream(:,:,randSegIdx) = featInStream;    % randomlize the patches
            for j=1:minibatch.nBatch
                minibatch.data{i,j} = featInStream(:,:, nSeqInMB*(j-1)+1 : min(nSeqInStreams,nSeqInMB*j));
            end
        end
    end
elseif para.NET.sentenceMinibatch == 1      % use sentences as minibatch
    minibatch.nBatch = size(feat,2);
    for i=1:nStream
        if isVAD(i); continue; end
        minibatch.data(i,:) = feat(i,:);
    end
else                                        % use randomly ordered frames as minibatch
    nFr = cellfun('size', feat(1,:),2);   % get the number of frames in each utterance
    % generate random frame index
    randFrameIdxInBlock = randperm(sum(nFr));   % if we don't use sentence minibatch, all data streams must be synchronized. It is enough to just use the first stream to compute number of frames
    for i=1:nStream
        if isVAD(i); continue; end
        feat_random{i} = cell2mat(feat(i,:));
        feat_random{i} = feat_random{i}(:, randFrameIdxInBlock);
    end
    total_nFr = sum(nFr);
    minibatch.nBatch = ceil(total_nFr/para.NET.batchSize);
    for i=1: minibatch.nBatch
        idx1 = (i-1)*para.NET.batchSize+1;
        idx2 = min(total_nFr, i*para.NET.batchSize);
        for j=1:length(feat_random)
            minibatch.data{j,i} = feat_random{j}(:, idx1:idx2);
        end
    end

%     for i=1:nStream
%         if isVAD(i); continue; end
%         minibatch.data{i} = cell2mat(feat(i,:));
%     end
%     if para.NET.sentenceMinibatch == 0
%         % generate random frame index
%         randFrameIdxInBlock = randperm(size(minibatch.data{1},2));   % if we don't use sentence minibatch, all data streams must be synchronized. It is enough to just use the first stream to compute number of frames
%         for i=1:length(minibatch.data)
%             minibatch.data{i} = minibatch.data{i}(:, randFrameIdxInBlock);
%         end
%     end
%     % generate the starting and ending frame index of each minibatch
%     if para.NET.sentenceMinibatch == 1  % in sequential training, we update the network per training utterance
%         minibatch.nBatch = size(feat,2);
%         for i=1:minibatch.nBatch
%             for j=1:size(feat,2)
%                 minibatch.idx1(i,j) = sum(nFr(1:i-1,j))+1;
%                 minibatch.idx2(i,j) = sum(nFr(1:i,j));
%             end
%         end
%     else
%         total_nFr = size(minibatch.data{1},2);
%         minibatch.nBatch = ceil(total_nFr/para.NET.batchSize);
%         for i=1:minibatch.nBatch
%             minibatch.idx1(i) = (i-1)*para.NET.batchSize+1;
%             minibatch.idx2(i) = min(total_nFr, i*para.NET.batchSize);
%         end
%         minibatch.idx1 = repmat(minibatch.idx1', 1, length(minibatch.data));
%         minibatch.idx2 = repmat(minibatch.idx2', 1, length(minibatch.data));
%     end
%     
%     if para.useGPU  % Move current block to GPU memory; remember to clear the memory when done
%         for i=1:length(minibatch.data)
%             if issparse(minibatch.data{i})
%                 if strcmp(datatype, 'single');
%                     nonzero_idx = find(sum(minibatch.data{i},2)>0);
%                     tmp_visible = full(minibatch.data{i}(nonzero_idx,:));
%                     minibatch.data{i} = zeros(visible_size, size(tmp_visible,2), datatype);
%                     minibatch.data{i}(nonzero_idx,:) = single(tmp_visible);
%                 else
%                     minibatch.data{i} = full(minibatch.data{i});
%                 end
%             end
%             minibatch.data{i} = gpuArray(minibatch.data{i});
%         end
%     end
end
end
