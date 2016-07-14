
function grad = B_log(future_layers, input,b)
future_grad = GetFutureGrad(future_layers, {});

grad = 1./(input+b).*future_grad;


end