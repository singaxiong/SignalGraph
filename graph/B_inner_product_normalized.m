function [grad] = B_inner_product_normalized(input_layers, future_layers)

future_grad = GetFutureGrad(future_layers, curr_layer);

input1 = input_layers{1}.a;
input2 = input_layers{2}.a;
dim = size(input1,1);

grad{1} = bsxfun(@times, input2/dim, future_grad);
grad{2} = bsxfun(@times, input1/dim, future_grad);

end
