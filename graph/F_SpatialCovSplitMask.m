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
if windowSize == 0      % utterance mode, estimate two spatial covariance matrixes for each utterance, one is speech and the other is noise.
    if N>1
        [validMask, variableLength] = getValidFrameMask(prev_layers{1});
        maskSpeechUtt = ExtractVariableLengthTrajectory(maskSpeech, validMask);
        maskNoiseUtt = ExtractVariableLengthTrajectory(maskNoise, validMask);
        dataUtt = ExtractVariableLengthTrajectory(data, validMask);
        for i=1:N
            output{i} = GetSCMUtt(dataUtt{i}, maskSpeechUtt{i}, maskNoiseUtt{i});
        end
        output = cell2mat_gpu(output);
        output = permute(output, [1 3 2]);
    else
        output = GetSCMUtt(data, maskSpeech, maskNoise);
    end
else        % online mode, estiamte covariance matrices for a sliding window of frames. 
    % to be implemented.    
end
end

%% 
function output = GetSCMUtt(data, maskSpeech, maskNoise)
[D,T,N] = size(maskSpeech);
[D2,T,N] = size(data);
nCh = D2/D;

data = reshape(data, D, nCh, T);
data = permute(data, [2 3 1]);

maskSpeech2 = permute(maskSpeech, [4 2 1 3]);
scm_speech = ComputeCovMask(data, maskSpeech2);

maskNoise2 = permute(maskNoise, [4 2 1 3]);
scm_noise = ComputeCovMask(data, maskNoise2);

scm_speech2 = reshape(scm_speech, nCh^2*D, 1, N);
scm_noise2 = reshape(scm_noise, nCh^2*D, 1, N);
output = [scm_speech2; scm_noise2];
end
