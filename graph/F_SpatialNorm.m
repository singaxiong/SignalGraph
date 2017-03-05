function normCovMat = F_SpatialNorm(prev_layer, curr_layer)

covMat = prev_layer.a;
prev_mask = prev_layer.validFrameMask;
nCh = curr_layer.nCh;
nBin = curr_layer.nBin;
[~, nf, N] = size(covMat);

if N == 1
    
    % normalize the cov matrix by their diagonal elements, remove the effect of
    % spectral power and only retains the phase information
    dimSelectMask1 = bsxfun(@times, eye(nCh, nCh), ones(nCh, nCh, nBin));
    dimSelectIdx1 = find(reshape(dimSelectMask1, numel(dimSelectMask1),1) == 1); % diag elements index
    diag_mean = squeeze(mean(reshape(covMat(dimSelectIdx1,:), nCh, nBin, nf), 1));
    if nf ==1
        diag_mean = diag_mean.';
    end
    normCovMat = bsxfun(@times, permute(reshape(covMat, nCh, nCh, nBin, nf), [3 4 1 2]), 1./diag_mean);
    normCovMat = reshape(permute(normCovMat, [3 4 1 2]), nCh^2*nBin, nf);
    
else
    % normalize the cov matrix by their diagonal elements, remove the effect of
    % spectral power and only retains the phase information
    dimSelectMask1 = bsxfun(@times, eye(nCh, nCh), ones(nCh, nCh, nBin));
    dimSelectIdx1 = find(reshape(dimSelectMask1, numel(dimSelectMask1),1) == 1); % diag elements index
    diag_mean = squeeze(mean(reshape(covMat(dimSelectIdx1,:,:), nCh, nBin, nf, N), 1));
    if nf ==1
        diag_mean = reshape(diag_mean, size(diag_mean,1), 1, size(diag_mean, 2));
    end
    % minibatch padding makes some frames zero, mean of that still be zero, can not be divided.
    diag_mean1 = permute(bsxfun(@plus, permute(diag_mean, [2 3 1]), -1e10.*prev_mask), [3 1 2]);
    normCovMat = bsxfun(@times, permute(reshape(covMat, nCh, nCh, nBin, nf, N), [3 4 5 1 2]), 1./diag_mean1);
    normCovMat = reshape(permute(normCovMat, [4 5 1 2 3]), nCh^2*nBin, nf, N);
    
end

end
