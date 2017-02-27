% compute the covariance matrix of data weighted by a mask
%   data: a 4D tensor (D1xD2xD3xD4), where D1 is the dimension of feature
%   vector. D2 is the number of vectors we are going to sum over. 
%   mask: a 3D tensor (D2xD3xD4), which specifies the weight of each
%   feature vector to the covariance matrix
%
function covMat = ComputeCovMask(data, mask)
[nCh, ~, nBin, N] = size(data);
weight = sqrt(bsxfun(@times, mask, 1./sum(mask)));
data_scaled = bsxfun(@times, data, weight);

% % version 1
% data_cell = num2cell(data_scaled, [1 2]);       % convert to cell array and call cellfun for speed
% tmp = cellfun(@(x) gather(x*x'), data_cell, 'UniformOutput', 0);
% covMat = cell2mat(tmp);
% % covMat = cell2mat_gpu(tmp);
% covMat = reshape(covMat, nCh^2*nBin, 1, N);

% version 2

covMat1 = outProdND(data_scaled);
covMat2 = squeeze(mean(covMat1, 3));
covMat = reshape(covMat2, nCh^2*nBin, 1, N);

end
