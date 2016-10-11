% repeat a matrix 
%
function [grad] = B_transpose(future_layers, curr_layer)
future_grad = GetFutureGrad(future_layers, curr_layer);

grad = future_grad';

end
