function feat = ExtractSpatialCovFeat(X, nCh, context_size, shift, binStep, rowOnly, nChLogMag)
if nargin<5;    binStep = 1; end
if nargin<6;    rowOnly = 0; end
if nargin<7;    nChLogMag = 7; end

[DC, T] = size(X);
nFFT = DC/nCh;
X2 = reshape(X, nFFT, nCh, T);

if 0    % apply Mel filterbank
    nFbank = 40;
    MelWindow = mel_window_FE(nFbank, nFFT-1, 16000)';
    MelWindow(:,end+1) = 0;
    X_tmp = reshape(X2, nFFT, nCh*T);
    X_fbank = MelWindow * X_tmp;
    data = reshape(X_fbank, nFbank, nCh, T);
else
    data = X2;
end

nBin = size(data,1);
bin_selector = binStep+1:binStep:nBin-1;    % we don't want the first and last bin as they are always real values
nSelected = length(bin_selector);

% compute the spatial covariance using context
if strcmpi(class(data), 'gpuArray')
    useGPU = 1;
else
    useGPU = 0;
end
spatialCov = ComplexSpectrum2SpatialCov(data(bin_selector,:,:), context_size, shift, useGPU);

% normalize the covariance matrix by their diagonal elements. This step
% removes the effect of spectral power and only retains the phase
% information (ideally). 
nFrCov = size(spatialCov,4);
if 0
    for i=1:nSelected
        for j=1:nFrCov
            meanPower(i,j) = mean(diag(spatialCov(:,:,i,j)));
        end
    end
elseif 0    % still slow
    spatialCovCell = num2cell(gather(spatialCov), [1 2]);
    meanPowerCell = cellfun(@(x) mean(diag(x)), spatialCovCell, 'UniformOutput', 0);
    meanPower = squeeze(cell2mat(meanPowerCell));
else    % use indexing
    dimSelectMask = zeros(nCh,nCh,nSelected);
    for i=1:nSelected
        dimSelectMask(:,:,i) = eye(nCh);
    end
    dimSelectIdx = find(dimSelectMask(:) == 1);
    spatialCov2 = reshape(spatialCov, nCh^2*nSelected,nFrCov);
    diag_part = spatialCov2(dimSelectIdx,:);
    meanPower = squeeze(mean(reshape(diag_part, nCh, nSelected, nFrCov)));
end
spatialCovNorm = bsxfun(@times, permute(spatialCov, [3 4 1 2]), 1./meanPower);
spatialCovNorm = permute(spatialCovNorm, [3 4 1 2]);


% get the upper triangle off-diagonal elements which are complex-valued
dimSelectMask = zeros(nCh,nCh,nSelected);
if rowOnly
    dimSelectMask(1,2:end,:) = 1;
else
    for i=1:nCh
        dimSelectMask(i,i+1:end,:) = 1;
    end
end
dimSelectIdx = find(dimSelectMask(:) == 1);

spatialCovNorm2 = reshape(spatialCovNorm, nCh^2*nSelected,nFrCov);
real_part = real(spatialCovNorm2(dimSelectIdx,:));
imag_part = imag(spatialCovNorm2(dimSelectIdx,:));

% get the diagoanl elements which are real values
dimSelectMask = zeros(nCh,nCh,nSelected);
for i=1:nSelected
    for j=1:nChLogMag
        dimSelectMask(j,j,i) = 1;
    end
end
dimSelectIdx = find(dimSelectMask(:) == 1);
spatialCov2 = reshape(spatialCov, nCh^2*nSelected,nFrCov);
diag_part = spatialCov2(dimSelectIdx,:);
diag_part = log(max(eps,abs(diag_part)));

% get the final feature vector
feat = [real_part; imag_part; diag_part];

end

