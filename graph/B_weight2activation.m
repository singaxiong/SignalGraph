function [grad, grad_W] = B_weight2activation(curr_layer, future_layers)

if isfield(curr_layer, 'isComplex')
    isComplex=curr_layer.isComplex;
else
    isComplex = 0;
end

grad = GetFutureGrad(future_layers, curr_layer);
if ~isComplex       % if the weight is not allowed to be complex, just take the real part
    grad = real(grad);
end

grad_W = grad;

if isfield(curr_layer, 'mask')
    grad_W = grad_W .* curr_layer.mask;
end

end
