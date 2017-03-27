function [grad] = B_softmax(future_layers, curr_layer)
future_grad = GetFutureGrad(future_layers, curr_layer);
output = curr_layer.a;

grad = future_grad .* output;

future_grad_output = sum(future_grad.*output);

grad2 = bsxfun(@times, output, future_grad_output);
grad = grad - grad2;

end
