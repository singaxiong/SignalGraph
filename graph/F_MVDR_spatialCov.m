% Estimate spatial covariance matrix for sentences using a mask. The mask
% specifies speech presense probability at all time frequency locations,
% with a 1 means speech present and 0 means speech absent. 
%
function curr_layer = F_MVDR_spatialCov(input_layer, curr_layer)
input = input_layer.a;
fs = curr_layer.fs;
freqBin = curr_layer.freqBin;
nFreqBin = length(freqBin);

[D,T,N] = size(input);
if isfield(curr_layer, 'noiseCovL2')
    noiseCovL2 = curr_layer.noiseCovL2;
else
    noiseCovL2 = 0;  % add noiseCovRegularization*\lambda*I to noise covariance, where \lambda is the maximum eigenvalue
end

D = D/2;
speechCov = input(1:D,:,:,:);
noiseCov = input(D+1:end,:,:,:);

dimTmp = size(speechCov,1) / nFreqBin;
nCh = sqrt(dimTmp);

speechCov = reshape(speechCov, nCh, nCh, nFreqBin, T, N);
noiseCov = reshape(noiseCov, nCh, nCh, nFreqBin, T, N);

speechCov_cell = num2cell(speechCov, [1 2]);       % convert to cell array and call cellfun for speed
noiseCov_cell = num2cell(noiseCov, [1 2]); 

% add regularization
if noiseCovL2 > 0
    eig_val = cellfun(@GetEigVal, noiseCov_cell, 'UniformOutput', 0);
    eig_val = cell2mat(eig_val);
    noise_floor = eig_val(end,:,:) * noiseCovL2;    
    % because our noise floor depends on noise covariance and network
    % parameters, it will not be able to pass through gradient check. This
    % is because we didn't consider this dependancy in the backpropagation.
    % But this should not have much effect on the training. 
    noise_floor_cell = num2cell(noise_floor, 1);
    noiseCov_cell = cellfun(@DiagLoading, noiseCov_cell, noise_floor_cell, 'UniformOutput', 0);
end

ninv_x = cellfun(@(x,n) (inv( n )*x), speechCov_cell, noiseCov_cell, 'UniformOutput', 0);
lambda = cellfun(@(x) abs(trace(x)), ninv_x, 'UniformOutput', 0);
if 0    % only lambda
    weight = cellfun(@(x,y) ones(size(x,1),1)/y, ninv_x, lambda, 'UniformOutput', 0);
elseif 0    % no lambda
    weight = cellfun(@(x) x(:,1), ninv_x, 'UniformOutput', 0);
else
    weight = cellfun(@(x,y) x(:,1)/y, ninv_x, lambda, 'UniformOutput', 0);
end
output = cell2mat(weight);
output = permute(output, [3 1 2 4 5]);
output = reshape(output, nFreqBin*nCh, T, N);

curr_layer.a = output;
curr_layer.lambda = lambda;
curr_layer.phi_s = speechCov_cell;
curr_layer.phi_n = noiseCov_cell;
if noiseCovL2>0
    curr_layer.noise_floor = noise_floor_cell;
end
end

%% 
function ev = GetEigVal(A)
[~,V] = eig(A);
ev = diag(V);
end

function B = DiagLoading(A, alpha)
D = size(A,1);
B = A + alpha * eye(D);
end
