% Estimate spatial covariance matrix for sentences using a speechMask. The speechMask
% specifies speech presense probability at all time frequency locations,
% with a 1 means speech present and 0 means speech absent. 
%
function [output, uttMask] = F_SpatialCovMask(prev_layers, curr_layer)
speechMask = prev_layers{1}.a;
data = prev_layers{2}.a;
prev_mask =prev_layers{2}.validFrameMask;

[D,T,N] = size(speechMask);
[D2,T,N] = size(data);
nCh = D2/D;
data = reshape(data, D, nCh, T, N);
data = permute(data, [2 3 1 4]);
% data = abs(data);

if isfield(curr_layer, 'winSize') && T > curr_layer.winSize
    windowSize = curr_layer.winSize;
    windowShift = curr_layer.winShift;
else
    windowSize = 0;
end

if isfield(curr_layer, 'speechOnly')
    speechOnly = curr_layer.speechOnly;
else
    speechOnly = false;
end

% if windowSize == 0
%     nf = 1;
% else
%     nf = fix((T-windowSize+windowShift)/windowShift);
% end
% uttMask = zeros(nf, N, 'like', real(data(1)));

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
                    scm_speech(:,:,d,n) = scm_speech(:,:,d,n) + speechMask(d,t,n) * data(:,t,d) * data(:,t,d)';
                    scm_noise(:,:,d,n) = scm_noise(:,:,d,n) + (1-speechMask(d,t,n)) * data(:,t,d) * data(:,t,d)';
                end
                scm_speech(:,:,d,n) = scm_speech(:,:,d,n) / sum(speechMask(d,:,n));
                scm_noise(:,:,d,n) = scm_noise(:,:,d,n) / (T-sum(speechMask(d,:,n)));
            end
        end
    else
%         vectorized: version 1
%         data_cell = num2cell(data, [1]);
%         mask_cell = num2cell(permute(speechMask, [3 2 1]), [1]);
%         scm_speech_cell = cellfun(@(x,y) (reshape(x*y*y',nCh^2,1)), mask_cell, data_cell, 'UniformOutput', 0);
%         scm_noise_cell = cellfun(@(x,y) (reshape((1-x)*y*y',nCh^2,1)), mask_cell, data_cell, 'UniformOutput', 0);
%         scm_speech = reshape(sum(cell2mat(scm_speech_cell),2),nCh,nCh,D);
%         scm_speech = bsxfun(@times, scm_speech, permute(1./sum(speechMask,2), [3 2 1]));
%         scm_noise = reshape(sum(cell2mat(scm_noise_cell),2),nCh,nCh,D);
%         scm_noise = bsxfun(@times, scm_noise, permute(1./sum(1-speechMask,2), [3 2 1]));
%         
        % version 2
        mask2 = permute(speechMask, [4 2 1 3]);
        scm_speech = ComputeCovMask(data, mask2);
        if speechOnly
            output = scm_speech;
        else
            scm_noise = ComputeCovMask(data, 1-mask2);
            output = [scm_speech; scm_noise];
        end
    end
    uttMask = zeros(1, N, 'like', real(data(1)));
else        % online mode, estiamte covariance matrices for a sliding window of frames.
    % to be implemented.
    % frame number after moving window
    mask2 = permute(speechMask, [4 2 1 3]);
    [scm_speech, uttMask] = ComputeWinCovMask(data, mask2, prev_mask, windowSize, windowShift);
    if speechOnly
        output = scm_speech;
    else
        scm_noise = ComputeWinCovMask(data, 1-mask2, prev_mask, windowSize, windowShift);
        output = [scm_speech; scm_noise];
    end
end

end
