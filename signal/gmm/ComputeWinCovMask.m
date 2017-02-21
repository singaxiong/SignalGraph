function winCovMat = ComputeWinCovMask(data, mask, winsize, shift)
[nch, nf_stft, nbin, N] = size(data);
weight = sqrt(bsxfun(@times, mask, 1./sum(mask)));
data_scaled = bsxfun(@times, data, weight);

covMat = outProdND(data_scaled);
if N == 1
    covMat1 = reshape(permute(covMat, [1 2 4 3]), nch^2*nbin, nf_stft);
else
    covMat1 = reshape(permute(covMat, [1 2 4 3 5]), nch^2*nbin, nf_stft, N);
end

% % Version 1: fast, but consume memory when winsize is large
% nf = fix((nf_stft-winsize+shift)/shift);
% covMat2 = ExpandContext_v2(covMat1, 0:winsize-1);
% nf_idx = 1:shift:nf_stft-winsize+1;
% covMat3 = covMat2(:, nf_idx, :);
% covMat3 = reshape(covMat3, nch^2*nbin, winsize, nf, N);
% winCovMat = squeeze(mean(covMat3, 2));

% Version 2: less fast than version 1
SCM1 = conv2(covMat1, ones(1,winsize, class(gather(covMat)))/winsize, 'valid');
winCovMat = SCM1(:, 1:shift:end);

% % Version 3: slowest in repmat and not support multiple sentences
% if IsInGPU(data)
%     winCovMat11 = gpuArray.zeros(nf, nch*nch*nbin*winsize);
% else
%     winCovMat11 = zeros(nf, nch*nch*nbin*winsize);
% end
% covMat11 = reshape(permute(covMat, [1 2 4 3]), 1, nch*nch*nbin*nf_stft);
% indf = nch*nch*nbin*shift*(0:(nf-1)).';
% inds = (1:nch*nch*nbin*winsize);
% % winCovMat(:) = covMat(indf(:,ones(1,nch*nch*nbin*winsize))+inds(ones(nf,1),:)); % slow
% winCovMat11(:) = covMat11(repmat(indf,1,nch*nch*nbin*winsize)+repmat(inds,nf,1));
% winCovMat11 = permute(reshape(winCovMat11, nf, nch*nch*nbin, winsize), [2 3 1]);
% winCovMat = squeeze(mean(winCovMat11, 2));

end