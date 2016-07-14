function [grad]= B_sigmoid(future_layers, output)
future_grad = GetFutureGrad(future_layers, []);
grad = future_grad .* output .* (1-output);

end