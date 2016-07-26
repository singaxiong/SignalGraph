function grad = B_beamforming(future_layers, prev_layers)
[input] =  prepareBeamforming(prev_layers);

future_grad = GetFutureGrad(future_layers, {});

input2 = permute(input, [1 3 2]);
grad = bsxfun(@times, conj(input2), future_grad);

end
