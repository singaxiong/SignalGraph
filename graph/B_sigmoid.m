function [grad]= B_sigmoid(future_layers, curr_layer)
output = curr_layer.a;
future_grad = GetFutureGrad(future_layers, curr_layer);
if isfield(curr_layer, 'rho')
    L1 = curr_layer.L1;
    L1weight = curr_layer.L1weight;
    rho = curr_layer.rho;
    
    tmp = -L1./max(1e-3,rho) + (1-L1)./max(1e-3,(1-rho));
    future_grad = future_grad + repmat(L1weight * tmp, 1, size(future_grad,2));
end

grad = future_grad .* output .* (1-output);



end