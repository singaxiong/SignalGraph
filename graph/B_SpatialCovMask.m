function grad = B_SpatialCovMask(future_layers, prev_layers, curr_layer)
mask = prev_layers{1}.a;
data = prev_layers{2}.a;
[D,T,N] = size(mask);
[D2,T,N] = size(data);
nCh = D2/D;
data = reshape(data, D, nCh, T, N);
data = permute(data, [2 3 1 4]);
% data = abs(data);

mask_f = sum(mask,2);     % sum of mask over time

spatCov = curr_layer.a;
speechCov = reshape(spatCov(1:D*nCh^2,:,:,:), nCh^2,1,D,N);
noiseCov = reshape(spatCov(D*nCh^2+1:end,:,:,:), nCh^2,1,D,N);

future_grad = GetFutureGrad(future_layers, curr_layer);
future_grad_speech = reshape(future_grad(1:D*nCh^2,:,:), nCh^2,1,D,N);
future_grad_noise = reshape(future_grad(D*nCh^2+1:end,:,:), nCh^2,1,D,N);

% from speech covariance
data_cell = num2cell(data, [1]); 
xx = cellfun(@(x) (reshape(x*x',nCh^2,1)), data_cell, 'UniformOutput', 0);
xx2 = cell2mat(xx);

gradFromSpeech = bsxfun(@minus, xx2, speechCov);
gradFromSpeech = bsxfun(@times, gradFromSpeech, future_grad_speech);
gradFromSpeech = squeeze(sum(gradFromSpeech))';
gradFromSpeech = bsxfun(@times, gradFromSpeech, 1./mask_f);

gradFromNoise = bsxfun(@minus, xx2, noiseCov);
gradFromNoise = bsxfun(@times, gradFromNoise, future_grad_noise);
gradFromNoise = squeeze(sum(gradFromNoise))';
gradFromNoise = bsxfun(@times, gradFromNoise, 1./(T-mask_f));

grad = gradFromSpeech - gradFromNoise;
% we may get complex valued gradient for real valued mask, as the partial
% derivative formula does not know that the mask must be real valued. To
% preserve the real-valued property of the mask, we just take the real part
% of the gradient, which represents the search direction of mask in the
% real domain.
grad = real(grad);      

end