function winCovMat = ComputeWinCovMask(data, mask, winsize, shift)

weight = sqrt(bsxfun(@times, mask, 1./sum(mask)));
data_scaled = bsxfun(@times, data, weight);

covMat = outProdND(data_scaled);

% convert the stft frame based Cov to winsize based Cov
[nch, ~, nf_stft, nbin] = size(covMat);
nf = fix((nf_stft-winsize+shift)/shift);
winCovMat = zeros(nf, nch*nch*nbin*winsize);
covMat = reshape(permute(covMat, [1 2 4 3]), 1, nch*nch*nbin*nf_stft);
indf = nch*nch*nbin*shift*(0:(nf-1)).';
inds = (1:nch*nch*nbin*winsize);
winCovMat(:) = covMat(indf(:,ones(1,nch*nch*nbin*winsize))+inds(ones(nf,1),:));
winCovMat = permute(reshape(winCovMat, nf, nch, nch, nbin, winsize), [2 3 4 5 1]);
winCovMat = squeeze(sum(winCovMat, 4)./winsize);
end