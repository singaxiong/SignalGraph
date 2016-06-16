% Givening a set of samples and their labels, we would like to divide the samples
% into N blocks randomly, such that each block will cover as much classes
% as possible and still respect the global distribution of the classes.
% Options:
%   refill: if set to 1, will refill minority classes that are used up. 
% 
% When call this function, make sure that the nSampleInBlock is larger than
% the number of classes. Otherwise, the sample index in each block may not
% follow the unigram closely. 
%
function sentIdx = AssignSamples2BlocksByLabel(label, nBlock, nSampleInBlock, refill, useLastBlock, DEBUG)
if nargin<6
    DEBUG = 0;
end
if nargin<5
    useLastBlock = 0;
end
if nargin<4
    refill = 0;
end

if refill
    minSampleNumPerClass = 1;   % minimum number of samples per class in a block
else
    minSampleNumPerClass = 0;
end

nTotalSample = length(label);
uniqueClass = unique(label);    % assume label is indexed from 1
nClass = length(uniqueClass);
% First, devide sentences into classes.
for i=1:nClass
    sampleInClass{i} = find(label==uniqueClass(i));
    classUnigram(i) = length(sampleInClass{i});
end
sampleInClass2 = sampleInClass;     % these two statistics will change in the following for loop
classUnigram2 = classUnigram;

for i=1:length(classUnigram2)
    sentIdxByClass{i} = [];     % for debug purpose only
end
sentIdx = {};
for bi = 1:nBlock
    nSampleInCurrBlock = min(nTotalSample-(bi-1)*nSampleInBlock, nSampleInBlock);
    if bi==nBlock && nSampleInCurrBlock < nSampleInBlock      % at the end of the loop, the remaining samples can be less than the required block size.
        % In this case, if refill is on, simply put all the remaining samples in the last block.
        % If refill is off, it's better to not use the last block as it may
        % have seriously biased class distribution. 
        if useLastBlock
            sentIdx{bi} = cell2mat(sampleInClass2); 
        end
        break;
    end
    % we first sample the class IDs for the current block
    classID4block = randperm(nClass);
    classID4block(nSampleInCurrBlock+1:end) = [];
    classID4block = sort(classID4block);
    % then decide how many samples we should get from each class for
    % current block
    nSampleInClass_block = DecideSampleNum4Class(classID4block, classUnigram2, nSampleInCurrBlock, minSampleNumPerClass);
   
    % then get samples from all classes
    sentIdxInBlock = {};
    for i = 1:length(classID4block)
        curr_classID = classID4block(i);
        randidx = randperm(classUnigram2(curr_classID));    % get a random index of the remaining samples in the current class
        selectedIdx = randidx(1:nSampleInClass_block(i));
        sentIdxInBlock{i} = sampleInClass2{curr_classID}(selectedIdx);
        % remove selected samples from the class
        sampleInClass2{curr_classID}(selectedIdx) = [];
        classUnigram2(curr_classID) = classUnigram2(curr_classID) - nSampleInClass_block(i);
        
        sentIdxByClass{curr_classID} = [sentIdxByClass{curr_classID} sentIdxInBlock{i}];
    end
    sentIdx{bi} = cell2mat(sentIdxInBlock);
    % check empty classes, i.e. classes whose samples are already used up.
    if refill
        for i=1:nClass
            if classUnigram2(i)==0
                % randomly get one sample from the class
                randidx = randperm(classUnigram(i));
                sampleInClass2{i} = sampleInClass{i}(randidx(1));
                classUnigram2(i) = 1;                   
            end
        end
    end
end
% check the unigram of each block
if DEBUG
    for i=1:length(sentIdx)
        for j = 1:nClass
            unigramRealized(i,j) = length(find(label(sentIdx{i})==j));
        end
    end
end
end



%% Decide how many sentences to sample from each classID
% We decide the number of samples in each class according to their unigram
% counts.
% Input:
%   class ID, an array of class IDs
%   classUnigram, the unigram count of the class IDs
%   nTotalSample, total number of samples we want to achieve for classes
%
function nSample = DecideSampleNum4Class(classID, classUnigram, nTotalSample, minSampleNumPerClass)
% first remove classes from classUnigram that does not appear in classID
classRequired = unique(classID);
classUnigram = classUnigram(classRequired);

nSample = [];
for ci = 1:length(classID)
    nRemainingSample2Assign = nTotalSample - sum(nSample(1:ci-1));
    remainingUnigram = classUnigram(ci:end);
    % decide the number of sentences in each class for current block based on the statitics
    nSample(ci) = max(minSampleNumPerClass,round(nRemainingSample2Assign/sum(remainingUnigram)*remainingUnigram(1)));
end
end