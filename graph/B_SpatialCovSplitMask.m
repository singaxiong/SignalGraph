function grad = B_SpatialCovSplitMask(future_layers, prev_layers, curr_layer)
maskSpeech = prev_layers{1}.a;
maskNoise = prev_layers{2}.a;
data = prev_layers{3}.a;

[D,T,N] = size(maskSpeech);
[D2,T,N] = size(data);
nCh = D2/D;
spatCov = curr_layer.a;
future_grad = GetFutureGrad(future_layers, curr_layer);

% data = abs(data);
if N>1
    [validMask, variableLength] = getValidFrameMask(curr_layer);
    maskSpeechUtt = ExtractVariableLengthTrajectory(maskSpeech, validMask);
    maskNoiseUtt = ExtractVariableLengthTrajectory(maskNoise, validMask);
    dataUtt = ExtractVariableLengthTrajectory(data, validMask);
    precision = class(gather(future_grad(1)));
    if IsInGPU(future_grad(1))
        grad{1} = gpuArray.zeros(D,T,N, precision);
        grad{2} = gpuArray.zeros(D,T,N, precision);
    else
        grad{1} = zeros(D,T,N, precision);
        grad{2} = zeros(D,T,N, precision);
    end
    for i=1:N
        gradTmp = GetGradUtt(dataUtt{i}, maskSpeechUtt{i}, maskNoiseUtt{i}, spatCov(:,:,i), future_grad(:,:,i));
        nFrSeg = size(dataUtt{i},2);
        grad{1}(:,1:nFrSeg,i) = gradTmp{1};
        grad{2}(:,1:nFrSeg,i) = gradTmp{2};
    end
else
    grad = GetGradUtt(data, maskSpeech, maskNoise, spatCov, future_grad);
end
end

%%
function grad = GetGradUtt(data, maskSpeech, maskNoise, spatCov, future_grad)
[D,T,N] = size(maskSpeech);
[D2,T,N] = size(data);
nCh = D2/D;

data = reshape(data, D, nCh, T, N);
data = permute(data, [2 3 1 4]);

maskSpeech_f = sum(maskSpeech,2);     % sum of mask over time
maskNoise_f = sum(maskNoise,2);     % sum of mask over time

speechCov = reshape(spatCov(1:D*nCh^2,:,:,:), nCh^2,1,D);
noiseCov = reshape(spatCov(D*nCh^2+1:end,:,:,:), nCh^2,1,D);

future_grad_speech = reshape(future_grad(1:D*nCh^2,:,:), nCh^2,1,D);
future_grad_noise = reshape(future_grad(D*nCh^2+1:end,:,:), nCh^2,1,D);

% from speech covariance
if 0    % slow implementation
    data = gather(data);
    data_cell = num2cell(data, [1]);
    tic; xx = cellfun(@(x) (reshape(x*x',nCh^2,1)), data_cell, 'UniformOutput', 0);toc;
    xx2 = cell2mat(xx);
else    % really fast implementation
    xx2 = reshape(outProdND(data), nCh^2, T,D);
end

gradFromSpeech = bsxfun(@minus, xx2, speechCov);
gradFromSpeech = bsxfun(@times, gradFromSpeech, future_grad_speech);
gradFromSpeech = squeeze(sum(gradFromSpeech))';
gradFromSpeech = bsxfun(@times, gradFromSpeech, 1./maskSpeech_f);

gradFromNoise = bsxfun(@minus, xx2, noiseCov);
gradFromNoise = bsxfun(@times, gradFromNoise, future_grad_noise);
gradFromNoise = squeeze(sum(gradFromNoise))';
gradFromNoise = bsxfun(@times, gradFromNoise, 1./maskNoise_f);

grad{1} = real(gradFromSpeech);
grad{2} = real(gradFromNoise);
% we may get complex valued gradient for real valued mask, as the partial
% derivative formula does not know that the mask must be real valued. To
% preserve the real-valued property of the mask, we just take the real part
% of the gradient, which represents the search direction of mask in the
% real domain.
end
