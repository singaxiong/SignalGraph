% Estimate spatial covariance matrix for sentences using a mask. The mask
% specifies speech presense probability at all time frequency locations,
% with a 1 means speech present and 0 means speech absent. 
%
function output = F_SpatialCovMask(prev_layers, curr_layer)
mask = prev_layers{1}.a;
data = prev_layers{2}.a;

if isfield(curr_layer, 'windowSize')
    windowSize = curr_layer.windowSize;
else
    windowSize = 0;
end

[D,T,N] = size(mask);
[D2,T,N] = size(data);
nCh = D2/D;
data = reshape(data, D, nCh, T, N);
data = permute(data, [2 3 1 4]);

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
                curr_mask = mask(d,:,n) ;
                curr_data = data(:, :, d, n);
                weighted_data = bsxfun(@times, curr_data, sqrt(curr_mask));
                scm_speech(:,:,d,n) = weighted_data * weighted_data' / sum(curr_mask);
                weighted_data = bsxfun(@times, curr_data, sqrt(1-curr_mask));
                scm_noise(:,:,d,n) = weighted_data * weighted_data' / (T-sum(curr_mask));
            end
        end
    else        % vectorized
        mask2 = permute(mask, [4 2 1 3]);
        scm_speech = ComputeCovMask(data, mask2);
        scm_noise = ComputeCovMask(data, 1-mask2);
    end
    
    scm_speech2 = reshape(scm_speech, nCh^2*D, 1, N);
    scm_noise2 = reshape(scm_noise, nCh^2*D, 1, N);
    output = [scm_speech2; scm_noise2];    
else        % online mode, estiamte covariance matrices for a sliding window of frames. 
    % to be implemented.    
end


end
