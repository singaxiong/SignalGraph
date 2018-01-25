% the class that holds the input streams of the signal graph. 
% each stream should either the data itself, or file name of the data. 
%   if use file names, should also have the file reader for the files. 
% this class should know 1) how to shuffle data; 2) how to prepare the next
% minibatch 

classdef GraphInput
    properties
        % configurations
        nStream = 0; %--- the number of input and output streams. E.g. if there is one input stream and one target for conventional supervised training, nStream=2. Default value is 2.
        randomizedBlock = 1;    % whether to randomize the sentences in the data streams. We may turn this off for CV data
        mode = 'normal';    % normal: ; dynamicPair: ; dynamicSimulation: simulate data on the fly for different iteratins
        
        % data generated by the functions
        streams = [];   % the actual data of the streams
        sentIdxInBlock = [];    % remembers the randomized sentence index for each data block
        nBlock = 0;     % number of blocks
        nMiniBatch = 0; 
    end
    methods
        function obj = GraphInput(nStream)
            obj.nStream = nStream;
        end
        
                
        function obj = GenIterationData(obj)
            switch lower(obj.mode)
                case 'normal'
                case 'dynamic_pair'
                    obj.streams = obj.GenDynamicPairs();
                case 'dynamic_simulation'
                    obj.streams = obj.GenDynamicDistortion();
                case 'dynamic_mixture'
                    obj.streams = obj.GenDynamicMixture();
            end
        end
        
        function streams = GenDynamicPairs(obj)
            
        end
        
        function streams = GenDynamicDistortion(obj)
            
        end
        
        function streams = GenDynamicMixture(obj)
            
        end
        
        function obj = ShuffleData(obj, para)
            % function [sentIdxInBlock] = shuffle_data(layer, para, data, isDev)
            if nargin<3;    isDev = 0; end
            % Decide the block size
            maxBlockSize = obj.decideMaxBlockSize(para.precision);
            %nSent = length(obj.data(1).data);
            rng('shuffle');
            
            nSent = length(obj.streams(1).data);
            nSentInBlock = min(maxBlockSize, para.NET.maxNumSentInBlock);
            obj.nBlock = ceil(nSent/double(nSentInBlock));
            
            % Assign sentences to blocks
            if obj.randomizedBlock==0
                randomSentIdx = 1:nSent;
            else
                randomSentIdx = randperm(nSent);
            end
            actual_nSentInBlock = [];
            for i=1:obj.nBlock
                actual_nSentInBlock(i) = min(nSentInBlock, nSent-sum(actual_nSentInBlock(1:i-1)));
                obj.sentIdxInBlock{i} = randomSentIdx( sum(actual_nSentInBlock(1:i-1))+1 : sum(actual_nSentInBlock(1:i)) );
            end
        end
        
        
        function minibatch = PrepareMinibatch(obj, dataType, sentenceBasedMinibatch, batchSize, blockIdx)
            currSentIdx = obj.sentIdxInBlock{blockIdx};
            nUtt = length(currSentIdx);
            feat = cell(obj.nStream, nUtt);
            for i = 1:obj.nStream   % for each input stream
                fprintf('Load data stream %d\n', i);
                feat(i,:) = obj.streams(i).getData(currSentIdx, dataType);
            end
            
            if sentenceBasedMinibatch      % use randomly ordered sequences as minibatch
                nSeqInMB = batchSize;
                nSeq = size(feat,2);
                obj.nMiniBatch = ceil(nSeq/nSeqInMB);
                randSegIdx = randperm(nSeq);
                for i=1:obj.nStream
                    featInStream = cell2mat_tensor3D(feat(i,:));
                    featInStream(:,:,randSegIdx) = featInStream;    % randomlize the patches
                    for j=1:obj.nMiniBatch
                        minibatch{i,j} = featInStream(:,:, nSeqInMB*(j-1)+1 : min(nSeq,nSeqInMB*j));
                    end
                end
            else                                        % use randomly ordered frames as minibatch
                nFr = cellfun('size', feat(1,:),2);   % get the number of frames in each utterance
                % generate random frame index
                randFrameIdxInBlock = randperm(sum(nFr));   % if we don't use sentence minibatch, all data streams must be synchronized. It is enough to just use the first stream to compute number of frames
                for i=1:obj.nStream
                    if isVAD(i); continue; end
                    feat_random{i} = cell2mat(feat(i,:));
                    feat_random{i} = feat_random{i}(:, randFrameIdxInBlock);
                end
                total_nFr = sum(nFr);
                obj.nMiniBatch = ceil(total_nFr/batchSize);
                for i=1: obj.nMiniBatch
                    idx1 = (i-1)*batchSize+1;
                    idx2 = min(total_nFr, i*batchSize);
                    for j=1:length(feat_random)
                        minibatch{j,i} = feat_random{j}(:, idx1:idx2);
                    end
                end
            end
        end
        
    end
    methods (Access = protected)
        function maxBlockSize = decideMaxBlockSize(obj,precision)
            % set the amount of memory we can use to store the block data
            [~,systemview] = memory;
            max_memory = systemview.PhysicalMemory.Available;
            max_memory = min(5e9, max_memory*0.8);
            
            for j=1:obj.nStream
                apprxSampleSize(j) = obj.streams(j).getApprxSampleSize();
            end
            
            max_sample = max_memory / sum(apprxSampleSize);
            
            if strcmpi(precision, 'single')
                maxBlockSize = max_sample / 4;   % use only half of the memory
            else
                maxBlockSize = max_sample / 8;
            end

        end
    end
end
