function [grad]= B_linear(future_layers, curr_layer)
grad = GetFutureGrad(future_layers, curr_layer);
end