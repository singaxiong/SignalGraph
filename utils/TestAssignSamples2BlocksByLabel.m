% test Assign

function TestAssignSamples2BlocksByLabel()

unigram = [5 9 20 1 8 7];
nSampleInBlock = 5;

label = [];
for i=1:length(unigram)
    label = [label ones(1,unigram(i))*i];
end
label = label(randperm(length(label)));
nBlock = ceil(length(label)/nSampleInBlock);

refill = 0;
DEBUG=1;
AssignSamples2BlocksByLabel(label, nBlock, nSampleInBlock, refill, DEBUG);

end
