function [grad, grad_W, grad_b] = B_beamforming_gainTDOA(future_layers, input_layer, curr_layer)
input = input_layer.a;
[D,T,N] = size(input);      

gain = curr_layer.W;   % W is a 2*nCh x nBin matrix. The first half store the real parts, the second half store the imaginary parts
TDOA = curr_layer.b;
[nCh,nBin] = size(gain);
freqBin = curr_layer.freqBin;

weight = F_tdoa2weight(TDOA(2:end), freqBin);
if isfield(curr_layer, 'useGain') && curr_layer.useGain==1
    weight = bsxfun(@times, weight, gain');
end

future_grad = GetFutureGrad(future_layers, curr_layer);
future_grad = reshape(future_grad,nBin,1,T,N);
input2 = reshape(input, nBin, nCh, T, N);
grad_Weight = bsxfun(@times, input2, future_grad);
grad_Weight = sum(reshape(grad_Weight,nBin,nCh,T*N),3);

% fakeCurrLayer.W = [real(weight) imag(weight)]';
% [grad, grad_W] = B_beamforming_freeWeight(future_layers, input_layer, fakeCurrLayer);
% grad_W = grad_W(1:nCh,:) + grad_W(nCh+1:end,:)*sqrt(-1);
% grad_W = grad_W.';

fakeLayer{1}.name = 'abc';
fakeLayer{1}.grad = grad_Weight;
fakeCurrLayer2.a = weight;
fakeCurrLayer2.freqBin = freqBin;
grad_tau = B_tdoa2weight(fakeLayer, fakeCurrLayer2);
grad_b = grad_tau;
grad_b(1) = 0;

grad = [];
grad_W = zeros(size(gain));

end
