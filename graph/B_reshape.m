function grad = B_reshape(input_layer, future_layers, curr_layer)
input = input_layer.a;

future_grad = GetFutureGrad(future_layers, curr_layer);

dim = [curr_layer.dim];
[D,T,N] = size(input);
% [D2,D3,TN] = size(future_grad);


grad = reshape(future_grad, D,T,N);

end