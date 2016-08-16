function [grad]= B_sigmoid(future_layers, curr_layer)
output = curr_layer.a;
future_grad = GetFutureGrad(future_layers, curr_layer);
grad = future_grad .* output .* (1-output);

end