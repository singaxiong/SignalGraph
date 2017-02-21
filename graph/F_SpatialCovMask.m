% Estimate spatial covariance matrix for sentences using a mask. The mask
% specifies speech presense probability at all time frequency locations,
% with a 1 means speech present and 0 means speech absent. 
%
function output = F_SpatialCovMask(prev_layers, curr_layer)
mask = prev_layers{1}.a;
data = prev_layers{2}.a;

if isfield(curr_layer, 'windowSize')
    windowSize = curr_layer.windowSize;
    shift = fix(windowSize/2);
else
    windowSize = 0;
end

if isfield(curr_layer, 'speechOnly')
    speechOnly = curr_layer.speechOnly;
else
    speechOnly = false;
end

[D,T,N] = size(mask);
[D2,T,N] = size(data);
nCh = D2/D;
data = reshape(data, D, nCh, T, N);
data = permute(data, [2 3 1 4]);
% data = abs(data);

if T <= windowSize
    windowSize = 0;
end

if windowSize == 0      % utterance mode, estimate two spatial covariance matrixes for each utterance, one is speech and the other is noise.
    if 0    % for loop version
        if IsInGPU(data)
            scm_speech = gpuArray.zeros(nCh, nCh, D, N);
            scm_noise = gpuArray.zeros(nCh, nCh, D, N);
        else
            scm_speech = zeros(nCh, nCh, D, N);
            scm_noise = zeros(nCh, nCh, D, N);
        end    
        for d=1:D
            for n=1:N
                for t=1:T
                    scm_speech(:,:,d,n) = scm_speech(:,:,d,n) + mask(d,t,n) * data(:,t,d) * data(:,t,d)';
                    scm_noise(:,:,d,n) = scm_noise(:,:,d,n) + (1-mask(d,t,n)) * data(:,t,d) * data(:,t,d)';
                end
                scm_speech(:,:,d,n) = scm_speech(:,:,d,n) / sum(mask(d,:,n));
                scm_noise(:,:,d,n) = scm_noise(:,:,d,n) / (T-sum(mask(d,:,n)));
            end
        end
    else        % vectorized
%         data_cell = num2cell(data, [1]);
%         mask_cell = num2cell(permute(mask, [3 2 1]), [1]);
%         scm_speech_cell = cellfun(@(x,y) (reshape(x*y*y',nCh^2,1)), mask_cell, data_cell, 'UniformOutput', 0);
%         scm_noise_cell = cellfun(@(x,y) (reshape((1-x)*y*y',nCh^2,1)), mask_cell, data_cell, 'UniformOutput', 0);
%         scm_speech = reshape(sum(cell2mat(scm_speech_cell),2),nCh,nCh,D);
%         scm_speech = bsxfun(@times, scm_speech, permute(1./sum(mask,2), [3 2 1]));
%         scm_noise = reshape(sum(cell2mat(scm_noise_cell),2),nCh,nCh,D);
%         scm_noise = bsxfun(@times, scm_noise, permute(1./sum(1-mask,2), [3 2 1]));
        
        mask2 = permute(mask, [4 2 1 3]);
        scm_speech = ComputeCovMask(data, mask2);
        if ~speechOnly
            scm_noise = ComputeCovMask(data, 1-mask2);
        end
    end
    
    scm_speech2 = reshape(scm_speech, nCh^2*D, 1, N);
    if speechOnly
        output = scm_speech2;
    else
        scm_noise2 = reshape(scm_noise, nCh^2*D, 1, N);
        output = [scm_speech2; scm_noise2];
    end
else        % online mode, estiamte covariance matrices for a sliding window of frames. 
    % to be implemented.
    % frame number after moving window
    nf = fix((T-windowSize+shift)/shift);
    mask2 = permute(mask, [4 2 1 3]);
    scm_speech = ComputeWinCovMask(data, mask2, windowSize, shift);
    if speechOnly
        output = scm_speech;
    else
        scm_noise = ComputeWinCovMask(data, 1-mask2, windowSize, shift);
        output = [scm_speech; scm_noise];
    end
end


end
