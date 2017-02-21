function feat = F_ExtSpatialCovFeat(prev_layer, curr_layer)

covMat = prev_layer.a;
prev_mask = prev_layer.validFrameMask;
nCh = curr_layer.nCh;
nBin = curr_layer.nBin;
[~, nf, N] = size(covMat);
scm_select = curr_layer.scm_select;

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
    
    % get the upper triangle off-diagonal elements which are complex-valued
    if strcmpi(scm_select, 'uptriangle')
        selectMat = triu(ones(nCh, nCh),1); % 1. up-trialgle
    elseif strcmpi(scm_select, 'row')
        selectMat = zeros(nCh, nCh); selectMat(1,2:end) = ones(1, nCh-1); % 2. first row
    else
        fprintf('Error: unknown scm feature select type: %s', lower(scm_select))
    end
    
    dimSelectMask2 = bsxfun(@times, selectMat, ones(nCh, nCh, nBin));
    dimSelectIdx2 = find(reshape(dimSelectMask2, numel(dimSelectMask2),1) == 1);
    real_part = real(normCovMat(dimSelectIdx2,:));
    % imag_part = imag(normCovMat(dimSelectIdx2,:));
    % for freq bin 1 and 257, no imag part
    dimSelectMask3 = bsxfun(@times, selectMat, cat(3,zeros(nCh, nCh, 1), ones(nCh, nCh, nBin-2), zeros(nCh, nCh, 1)));
    dimSelectIdx3 = find(reshape(dimSelectMask3, numel(dimSelectMask3),1) == 1);
    imag_part = imag(normCovMat(dimSelectIdx3,:));
    
    % get the diagonal elements which are real values
    % diag_part = covMat(dimSelectIdx1,:);
    % diag_part = log(max(eps,abs(diag_part)));
    diag_part = real(normCovMat(dimSelectIdx1,:));
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
    
    % get the upper triangle off-diagonal elements which are complex-valued
    if strcmpi(scm_select, 'uptriangle')
        selectMat = triu(ones(nCh, nCh),1); % 1. up-trialgle
    elseif strcmpi(scm_select, 'row')
        selectMat = zeros(nCh, nCh); selectMat(1,2:end) = ones(1, nCh-1); % 2. first row
    else
        fprintf('Error: unknown scm feature select type: %s', lower(scm_select))
    end
    
    dimSelectMask2 = bsxfun(@times, selectMat, ones(nCh, nCh, nBin));
    dimSelectIdx2 = find(reshape(dimSelectMask2, numel(dimSelectMask2),1) == 1);
    real_part = real(normCovMat(dimSelectIdx2,:,:));
    % imag_part = imag(normCovMat(dimSelectIdx2,:));
    % for freq bin 1 and 257, no imag part
    dimSelectMask3 = bsxfun(@times, selectMat, cat(3,zeros(nCh, nCh, 1), ones(nCh, nCh, nBin-2), zeros(nCh, nCh, 1)));
    dimSelectIdx3 = find(reshape(dimSelectMask3, numel(dimSelectMask3),1) == 1);
    imag_part = imag(normCovMat(dimSelectIdx3,:,:));
    
    % get the diagonal elements which are real values
    diag_part = real(normCovMat(dimSelectIdx1,:,:));
end

% get the final feature vector
feat = [real_part; imag_part; diag_part];
% real_part = reshape(real_part, 7, 257, nf, N);
% imag_part = reshape(imag_part, 7, 255, nf, N);
% real_part = real_part(:, 6:5:end,:,:);
% imag_part = imag_part(:, 5:5:end,:,:);
% 
% feat = [reshape(real_part, 7*51, nf, N); reshape(imag_part, 7*51, nf, N)];


% covMat = reshape(covMat(:,:,:), nCh, nCh, nBin, nf, N);
% covMatCell = num2cell(covMat, [1 2]);
% omegaTau = cellfun(@GetPrincVec, covMatCell, 'UniformOutput', 0);
% output = permute(cell2mat(omegaTau), [1 3 4 5 2]);
% 
% feat = output(2:8, 5:5:end, :,:);
% [d1,d2,d3,d4] = size(feat);
% feat = reshape(feat, d1*d2, d3, d4);

end

function omegaTau = GetPrincVec(A)
[V,D] = eig(A);
D = diag(D);
[~, idx] = max(D);
ev = V(:,idx);
omegaTau = gather(angle(ev/ev(1)));
end
