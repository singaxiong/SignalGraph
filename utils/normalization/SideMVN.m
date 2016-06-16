function [meanV varV mfcMVN] = SideMVN(inputFiles, outputFiles)

mfcRaw = {};    nFr = [];
mfcMVN = {};
meanAcc = 0;    varAcc = 0;
for j=1:length(inputFiles)
    if strcmp(inputFiles{j}(end-2:end), 'mfc')
        mfcRaw{end+1} = readHTK(inputFiles{j}, 39 );
        nFr(length(mfcRaw)) = size(mfcRaw{end},1);
        meanAcc = meanAcc + sum(mfcRaw{end});
        varAcc = varAcc + sum(mfcRaw{end}.^2);
    end
end
meanV = meanAcc / sum(nFr);
varV = sqrt( varAcc/sum(nFr) - meanV.^2 );
% Direct but slow implementation
if 0
    mfcAcc = [];
    for j=1:length(mfcRaw)
        mfcAcc = [mfcAcc; mfcRaw{j}];
    end
    meanV2 = mean(mfcAcc);
    varV2 = var(mfcAcc);
end

% Normalize the mean and variance
for j=1:length(mfcRaw)
    mfcMVN{j} = ( mfcRaw{j} - repmat(meanV, nFr(j),1) ) ./ repmat(varV, nFr(j),1);
end

% Write the MVN processed features
cnt = 0;
for j=1:length(outputFiles)
    if strcmp(outputFiles{j}(end-2:end), 'mfc')
        cnt = cnt + 1;
        writeHTK(outputFiles{j}, mfcMVN{cnt}, 'MFCC_0_D_A');
    end
end

