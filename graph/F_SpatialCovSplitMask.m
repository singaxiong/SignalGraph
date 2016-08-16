% Estimate spatial covariance matrix for sentences using a mask. The mask
% specifies speech presense probability at all time frequency locations,
% with a 1 means speech present and 0 means speech absent. 
%
function output = F_SpatialCovSplitMask(prev_layers, curr_layer)
maskSpeech = prev_layers{1}.a;
maskNoise = prev_layers{2}.a;
data = prev_layers{3}.a;

if isfield(curr_layer, 'windowSize')
    windowSize = curr_layer.windowSize;
else
    windowSize = 0;
end

[D,T,N] = size(maskSpeech);
[D2,T,N] = size(data);
nCh = D2/D;
data = reshape(data, D, nCh, T, N);
data = permute(data, [2 3 1 4]);

if windowSize == 0      % utterance mode, estimate two spatial covariance matrixes for each utterance, one is speech and the other is noise.

    maskSpeech2 = permute(maskSpeech, [4 2 1 3]);
    scm_speech = ComputeCovMask(data, maskSpeech2);
    
    maskNoise2 = permute(maskNoise, [4 2 1 3]);    
    scm_noise = ComputeCovMask(data, maskNoise2);
    
    scm_speech2 = reshape(scm_speech, nCh^2*D, 1, N);
    scm_noise2 = reshape(scm_noise, nCh^2*D, 1, N);
    output = [scm_speech2; scm_noise2];    
else        % online mode, estiamte covariance matrices for a sliding window of frames. 
    % to be implemented.    
end


end
