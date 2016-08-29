function [grad, grad_W] = B_weight2activation(curr_layer, future_layers)

grad = GetFutureGrad(future_layers, curr_layer);
grad_W = grad;


end
