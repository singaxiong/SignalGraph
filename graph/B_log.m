
function grad = B_log(future_layers, input,curr_layer)
future_grad = GetFutureGrad(future_layers, curr_layer);

grad = 1./(input+curr_layer.const).*future_grad;


end