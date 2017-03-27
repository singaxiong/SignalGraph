function [grad] = B_ExtractDims(input_layer, curr_layer, future_layers)

input = input_layer{1}.a;

future_grad = GetFutureGrad(future_layers, curr_layer);

[D,T,N] = size(input);

precision = class(gather(future_grad(1,1,1)));
if strcmpi(class(future_grad), 'gpuArray')
    grad = gpuArray.zeros(D,T, N, precision);
else
    grad = zeros(D,T, N, precision);
end

grad(curr_layer.dimIndex,:,:) = future_grad;

end