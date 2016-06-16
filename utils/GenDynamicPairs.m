% This function dynamically generates training pairs during neural networks
% training. The advantage is that we can use different pairs in different
% epoch. So theoretically, we will use all the possible training pairs if
% the training runs long enough. 
%
% Use scenarios: given N samples belonging to C classes, generate M pairs for
% discriminate metric learning. Both positive pairs (same class) and
% negative pairs (different classes) are generated. 
% 
% Authors: Xiong Xiao, NTU, Singapore
% Last Modified: 1 Jun 2016
%
function [data_pair] = GenDynamicPairs(data, para)

% assign the base database to the two streams of pair. We will represent
% pairs as indexes of the bases to save space. 
data_pair(1).base = data(para.IO.dynamicPair.dataIdx).base;
data_pair(2).base = data_pair(1).base;

% get the class ID of the data
label = data(para.IO.dynamicPair.labelIdx).base;
nMinSamplePerClass = para.IO.dynamicPair.nMinSamplePerClass;   % minimum number of samples per class
ratio = para.IO.dynamicPair.ratio;      % ratio between positive and negative samples

vocab = unique(label);
for i=1:length(vocab)
    class_idx{i} = find(label==vocab(i));
    nSampleClass(i) = length(class_idx{i});
end
data_idx = 1:length(label);

% repeat the data for classes with very little data
for i=1:length(class_idx)
    curr_data_idx = data_idx(class_idx{i});
    if length(class_idx{i})<nMinSamplePerClass
        % repeat the data many times until the number of ivector is larger
        % than the minimum requirement
        augmented_data_idx = [curr_data_idx];
        while 1
            rand_idx_aug = randperm(size(curr_data_idx,2));
            augmented_data_idx = [augmented_data_idx curr_data_idx(:,rand_idx_aug)];
            if size(augmented_data_idx,2)>=nMinSamplePerClass
                break;
            end
        end
        data_idx2{i} = augmented_data_idx(:, 1:nMinSamplePerClass);
    else
        data_idx2{i} = curr_data_idx;
    end
end

offset = 0;
for i=1:length(data_idx2)
    class_idx2{i} = (1:size(data_idx2{i},2)) + offset;
    offset = offset + length(class_idx2{i});
end
data_idx2 = [data_idx2{:}];

% for each sample, we create 2 pairs, one is a positive pair that the
% samples belong to the same class, the other is a negative pair between
% different langauges of the same cluster

for i=1:length(vocab),
    PrintProgress(i, length(vocab), 500, 'GenDynamicPairs');
    % positive pairs
    nSampleInClass = length(class_idx2{i});
    randIdxPos1 = class_idx2{i}(randperm(nSampleInClass));
    randIdxPos2 = class_idx2{i}(randperm(nSampleInClass));
    pair_idx_pos{i} = [randIdxPos1(:) randIdxPos2(:)];
    
    % negative pairs
    % first, find the index that does not include the current class
    other_class_idx = class_idx2( [1:length(vocab)] ~= i);  % remove the current class from the idx
    other_class_idx = cell2mat(other_class_idx);
    
    % then generate random index
    nSampleInOtherClass = length(other_class_idx);
    nSample = min(ratio*nSampleInClass, nSampleInOtherClass);
    randIdxPos_overall = [];
    for j=1:ratio,
        randIdxPos_overall = [randIdxPos_overall class_idx2{i}(randperm(nSampleInClass))];
    end    
    randIdxNeg = randperm(nSampleInOtherClass, nSample);
    pair_idx_neg{i} = [randIdxPos_overall(1:nSample)' other_class_idx(randIdxNeg)'];
end
pair_idx_pos_all = cell2mat(pair_idx_pos');
pair_idx_neg_all = cell2mat(pair_idx_neg');
side1_idx = [pair_idx_pos_all(:,1); pair_idx_neg_all(:,1)];
side2_idx = [pair_idx_pos_all(:,2); pair_idx_neg_all(:,2)];
all_label = [ones(1, size(pair_idx_pos_all,1))*para.IO.dynamicPair.targets(1)  ones(1, size(pair_idx_neg_all,1))*para.IO.dynamicPair.targets(2) ];     % 1 for same class, 2 for different classes

if 0    % debug
    plot(pair_idx_pos_all(:,1), pair_idx_pos_all(:,2), '*'); hold on;
    plot(pair_idx_neg_all(:,1), pair_idx_neg_all(:,2), 'r+'); hold off;
end
    
% randomize the index
randidx = randperm(length(side1_idx));
side1_idx = side1_idx(randidx);
side2_idx = side2_idx(randidx);
all_label = all_label(randidx);

% divide the index into blocks
BLK_SIZE = 10000;
nBlock = ceil(length(side1_idx)/BLK_SIZE);

for i=1:nBlock
    start = (i-1)*BLK_SIZE+1;
    stop = min(length(side1_idx), i*BLK_SIZE);
    data_pair(1).data{i} = data_idx2(:,side1_idx(start:stop));
    data_pair(2).data{i} = data_idx2(:,side2_idx(start:stop));
    data_pair(3).data{i} = all_label(start:stop);
end
end
