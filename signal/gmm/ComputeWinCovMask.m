function [winCovMat, winMask] = ComputeWinCovMask(data, mask, prev_mask, windowSize, windowShift)
[nCh, nf_stft, nBin, N] = size(data);
weight = sqrt(bsxfun(@times, mask, 1./sum(mask)));
data_scaled = bsxfun(@times, data, weight);

covMat = outProdND(data_scaled);

nf = fix((nf_stft-windowSize+windowShift)/windowShift);
winMask = zeros(nf, N, 'like', real(covMat(1)));

if N == 1
    covMat1 = reshape(permute(covMat, [1 2 4 3]), nCh^2*nBin, nf_stft);
%     covMat1 = repmat(mean(covMat1,2), 1, size(covMat1, 2));
    % % Version 1: fast, but consume memory when windowSize is large
%     nf = fix((nf_stft-windowSize+windowShift)/windowShift);
%     covMat2 = ExpandContext_v2(covMat1, 0:windowSize-1);
%     nf_idx = 1:windowShift:nf_stft-windowSize+1;
%     covMat3 = covMat2(:, nf_idx, :);
%     covMat3 = reshape(covMat3, nCh^2*nBin, windowSize, nf, N);
%     winCovMat = squeeze(mean(covMat3, 2));
%     
    % Version 2: less fast than version 1
    SCM1 = conv2(covMat1, ones(1,windowSize, class(gather(covMat)))/windowSize, 'valid');
    winCovMat = SCM1(:, 1:windowShift:end);
%     
%     % Version 3: slowest in repmat and not support multiple sentences
%     if IsInGPU(data)
%         winCovMat11 = gpuArray.zeros(nf, nCh*nCh*nBin*windowSize);
%     else
%         winCovMat11 = zeros(nf, nCh*nCh*nBin*windowSize);
%     end
%     covMat11 = reshape(permute(covMat, [1 2 4 3]), 1, nCh*nCh*nBin*nf_stft);
%     indf = nCh*nCh*nBin*windowShift*(0:(nf-1)).';
%     inds = (1:nCh*nCh*nBin*windowSize);
%     % winCovMat(:) = covMat(indf(:,ones(1,nCh*nCh*nBin*windowSize))+inds(ones(nf,1),:)); % slow
%     winCovMat11(:) = covMat11(repmat(indf,1,nCh*nCh*nBin*windowSize)+repmat(inds,nf,1));
%     winCovMat11 = permute(reshape(winCovMat11, nf, nCh*nCh*nBin, windowSize), [2 3 1]);
%     winCovMat = squeeze(mean(winCovMat11, 2));
else
%     % version 1
%     covMat1 = reshape(permute(covMat, [1 2 4 3 5]), nCh^2*nBin, nf_stft, N);
%     winCovMat = zeros(nCh^2*nBin, nf, N, 'like', covMat1(1));
%     for i=1:N
%         idx = find(prev_mask(:,i) == 0, 1, 'last');
%         idx2 = fix((idx-windowSize+windowShift)/windowShift);
%         covMat2 = squeeze(covMat1(:,1:idx,i));
%         SCM = conv2(covMat2, ones(1,windowSize, 'like', covMat1(1))/windowSize, 'valid');
%         winCovMat(:, 1:idx2, i) = SCM(:, 1:windowShift:end);
%         winMask(idx2+1:end, i) = 1;
%     end
    
    % Version 2, much fast
    covMat2 = reshape(permute(covMat, [1 2 4 5 3]), nCh^2*nBin*N, nf_stft);
    idx = arrayfun(@(x) find(gather(prev_mask(:,x)) == 0, 1, 'last'), 1:size(prev_mask,2));
    idx2 = arrayfun(@(x) fix((idx(x)-windowSize+windowShift)/windowShift), 1:length(idx));
    covMat3 = conv2(covMat2, ones(1,windowSize, 'like', covMat2(1))/windowSize, 'valid');
    winCovMat1 = covMat3(:, 1:windowShift:end);
    winCovMat2 = permute(reshape(winCovMat1, nCh^2*nBin, N, size(winCovMat1, 2)), [1 3 2]);
    winCovMat = zeros(nCh^2*nBin, nf, N, 'like', winCovMat2(1));
    for i = 1:N
        winCovMat(:, 1:idx2(i), i) = winCovMat2(:, 1:idx2(i), i);
        winMask(idx2(i)+1:end, i) = 1;
    end
    
end
end