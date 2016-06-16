function grad = B_beamforming_power(future_layer, curr_layer, prev_layer)
% we don't implement this function now, as it will be used with
% beamforming. We will implement the joint gradient of Power spectrum and
% frequency domain beamforming. 
% X is the multichannel complex spectrum inputs
future_grad = future_layer{1}.grad;
[X, weight] =  prepareBeamforming(prev_layer);
[C,T,N] = size(X);
% Y is the beamforming's output
Y = curr_layer.a;

if 0
    for i=1:C
        cross_term = future_grad .* Y .* squeeze(conj(X(i,:,:)))';
        grad(:,i) = 2*sum(cross_term,2);
    end
else
    future_grad_Y = future_grad .* Y;
    X_tmp = permute(X, [3 2 1]);
    cross_term = bsxfun(@times, X_tmp, future_grad_Y);
    grad = 2*squeeze(sum(cross_term,2));
end

end
