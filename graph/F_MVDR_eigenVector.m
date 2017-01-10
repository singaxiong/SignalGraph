% Estimate steering vector using eigen-decomposition of speech PSD matrix. 
%
function curr_layer = F_MVDR_eigenVector(input_layer, curr_layer)
input = input_layer.a;
fs = curr_layer.fs;
freqBin = curr_layer.freqBin;
nFreqBin = length(freqBin);

[D,T,N] = size(input);
noiseCovRegularization = 0.1;

D = D/2;
speechCov = input(1:D,:,:,:);
noiseCov = input(D+1:end,:,:,:);

dimTmp = size(speechCov,1) / nFreqBin;
nCh = sqrt(dimTmp);

speechCov = reshape(speechCov, nCh, nCh, nFreqBin, T, N);
noiseCov = reshape(noiseCov, nCh, nCh, nFreqBin, T, N);

speechCov_cell = num2cell(speechCov, [1 2]);       % convert to cell array and call cellfun for speed
noiseCov_cell = num2cell(noiseCov, [1 2]); 

[u, v] = cellfun(@(x) eig(x), speechCov_cell, 'UniformOutput', 0);
weight = cellfun(@(x, n) inv(n)*x(:, end)/(x(:, end)'*inv(n)*x(:, end)), u, noiseCov_cell, 'UniformOutput', 0);
U = cell2mat(u);
V = cell2mat(v);

output = cell2mat(weight);
output = permute(output, [3 1 2]);
output = reshape(output, nFreqBin*nCh, T, N);

curr_layer.a = output;
curr_layer.phi_s = speechCov_cell;
curr_layer.phi_n = noiseCov_cell;
curr_layer.u = u;
curr_layer.v = v;

end
