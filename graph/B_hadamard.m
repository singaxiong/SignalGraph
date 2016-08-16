function [grad]= B_hadamard(input_layers, curr_layer, future_layers)

input = input_layers{1}.a;
mask = input_layers{2}.a;

future_grad = GetFutureGrad(future_layers, curr_layer);

[n1,n2,n3] = size(future_grad);

grad{1} = future_grad .* mask;
grad{2} = real(future_grad .* input);

end
