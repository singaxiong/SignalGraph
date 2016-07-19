% Estimate spatial covariance matrix for sentences using a mask. The mask
% specifies speech presense probability at all time frequency locations,
% with a 1 means speech present and 0 means speech absent. 
%
function output = F_MVDR_spatialCov(input_layer, curr_layer)
input = input_layer.a;
fs = curr_layer.fs;
freqBin = curr_layer.freqBin;
nFreqBin = length(freqBin);

[D,T,N] = size(input);

D = D/2;
speechCov = input(1:D,:,:,:);
noiseCov = input(D+1:end,:,:,:);

dimTmp = size(speechCov,1) / nFreqBin;
nCh = sqrt(dimTmp);

speechCov = reshape(speechCov, nCh, nCh, nFreqBin, T, N);
noiseCov = reshape(noiseCov, nCh, nCh, nFreqBin, T, N);

speechCov_cell = num2cell(speechCov, [1 2]);       % convert to cell array and call cellfun for speed
noiseCov_cell = num2cell(noiseCov, [1 2]); 

ninv_x = cellfun(@(x,n) (inv(n)*x), speechCov_cell, noiseCov_cell, 'UniformOutput', 0);
weight = cellfun(@(x) x(:,1), ninv_x, 'UniformOutput', 0);
lambda = cellfun(@(x) abs(trace(x)), ninv_x, 'UniformOutput', 0);
output = cellfun(@(x,y) x/y, weight, lambda, 'UniformOutput', 0);
output = cell2mat(output);
output = permute(output, [3 1 2]);
output = reshape(output, nFreqBin*nCh, T, N);

end
