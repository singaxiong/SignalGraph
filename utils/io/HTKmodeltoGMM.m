% This function convert a HTK monophone model file to a big GMM file
function model = HTKmodeltoGMM(modelName, phoneList)
phonemes = my_cat(phoneList);
hmms = readHTKmodel(modelName,phoneList);

model.meanC = [];
model.covC = [];
model.prior = [];

mixIdx = 0;
for i=1:length(phonemes)
    if strcmp(phonemes{i}, 'sp')==1
        continue;
    end
    if strcmp(phonemes{i},'@')
        phonemes{i} = 'AA';
    end
    for j=1:hmms.(phonemes{i}).NUMSTATES
        for k=1:hmms.(phonemes{i}).NUMMIXES(j)
            mixIdx = mixIdx+1;
            model.prior(mixIdx) = hmms.(phonemes{i}).weight(k,j);
            model.meanC(:,mixIdx) = hmms.(phonemes{i}).mean(:,k,j);
            model.covC(:,mixIdx) = hmms.(phonemes{i}).var(:,k,j);
        end
    end    
end

% model.prior = model.prior / sum(model.prior);
