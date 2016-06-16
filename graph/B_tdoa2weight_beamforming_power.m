function grad = B_tdoa2weight_beamforming_power(X, beamform_layer, after_power_layer, tdoa_layer)
% X is the multichannel complex spectrum inputs
[C,T,N] = size(X);
% Y is the beamforming's output
Y = beamform_layer.a;
% weight is the beamforming weight
weight = tdoa_layer.a;
% future_grad is the gradient of the power specturm of beamformed signal
future_grad = after_power_layer.grad;

omega = tdoa_layer.freqBin;

jw = sqrt(-1)*omega;
term1 = bsxfun(@times, weight, -jw.');
future_grad_Y = future_grad.*Y;
for c = 1:C
    
    cross_term_future_grad = future_grad_Y .* squeeze(X(c,:,:))';
    tmp = bsxfun(@times, cross_term_future_grad, term1(:,c));
    grad(c) = 2*sum(sum(  real( tmp )));
%    cross_term = Y .* squeeze(X(c,:,:))';
%    grad(c) = 2*sum(sum( future_grad .* real( repmat(conj(term1'),1,T) .* cross_term )));
end
grad = grad(2:end);
grad = repmat(grad(:), 1, T)/T;

end
