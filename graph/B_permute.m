% 
%
function [grad] = B_permute(future_layers, curr_layer)
future_grad = GetFutureGrad(future_layers, curr_layer);

%decide the permute order to undo the permutation of forward pass
permute_order = curr_layer.permute_order;
[~,reverse_permute_order] = sort(permute_order);

grad = permute(future_grad, reverse_permute_order);

end
