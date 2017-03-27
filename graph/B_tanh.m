function [grad]= B_tanh(future_layers, curr_layer)
output = curr_layer.a;
future_grad = GetFutureGrad(future_layers, curr_layer);
grad = (1-output.^2) .* future_grad;

end