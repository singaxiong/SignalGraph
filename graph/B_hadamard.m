function [grad]= B_hadamard(input_layers, curr_layer, future_layers)

input = input_layers{1}.a;
input2 = input_layers{2}.a;

future_grad = GetFutureGrad(future_layers, curr_layer);

[n1,n2,n3] = size(future_grad);

grad{1} = future_grad .* input2;
grad{2} = future_grad .* input;

end
