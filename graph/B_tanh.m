function [grad]= B_tanh(future_layers, output)
future_grad = GetFutureGrad(future_layers, []);
grad = (1-output.^2) .* future_grad;

end