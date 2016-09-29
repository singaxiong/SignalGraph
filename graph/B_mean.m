function grad = B_mean(input_layer, future_layers, curr_layer)
input = input_layer.a;
[D,T,N] = size(input);

future_grad = GetFutureGrad(future_layers, curr_layer);

poolIdx = curr_layer.pool_idx;
if poolIdx==3
    grad = repmat(future_grad/N, [1 1 N]);
elseif poolIdx==2
    grad = repmat(future_grad/T, [1 T 1]);
else
    grad = repmat(future_grad/D, [D 1 1]);    
end

end
