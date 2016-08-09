function [grad, grad_W] = B_beamforming_freeWeight(future_layers, input_layer, curr_layer)
input = input_layer.a;
[D,T,N] = size(input);      

W = curr_layer.W;   % W is a 2*nCh x nBin matrix. The first half store the real parts, the second half store the imaginary parts
[D2,nBin] = size(W);
nCh = D2/2;
Wcomplex = W(1:nCh,:) + W(nCh+1:end,:)*sqrt(-1);

input2 = reshape(input, nBin, nCh, T, N);

future_grad = GetFutureGrad(future_layers, curr_layer);
future_grad = reshape(future_grad,nBin,1,T,N);

grad_W = bsxfun(@times, input2, future_grad);
grad_W = sum(reshape(grad_W,nBin,nCh,T*N),3);
grad_W = grad_W.';
grad_W = [real(grad_W); imag(grad_W)];

grad = bsxfun(@times, future_grad, Wcomplex');
grad = reshape(grad, nBin*nCh, T, N);
grad = conj(grad);
end
