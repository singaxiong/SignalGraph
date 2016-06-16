function [mfcMVN, meanV, varV] = SideMVNcore(mfcRaw)

meanAcc = 0;    varAcc = 0;
for j=1:length(mfcRaw)
    nFr(j) = size(mfcRaw{j},1);
    meanAcc = meanAcc + sum(mfcRaw{j});
    varAcc = varAcc + sum(mfcRaw{j}.^2);
end
meanV = meanAcc / sum(nFr);
varV = sqrt( varAcc/sum(nFr) - meanV.^2 );
precision = 1./varV;

% Normalize the mean and variance
for j=1:length(mfcRaw)
    mfcMVN{j} = bsxfun(@minus, mfcRaw{j}, meanV);
    mfcMVN{j} = bsxfun(@times, mfcMVN{j}, precision);
end
