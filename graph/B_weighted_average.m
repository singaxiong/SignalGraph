function [grad, grad_W_raw] = B_weighted_average(prev_layers, curr_layer, future_layers)

% raw_weights = prev_layers{1}.a;
input = prev_layers{2}.a;
[dim, nFr] = size(input);

% raw_weights2 = exp(raw_weights);
% weights = raw_weights2 / sum(raw_weights2);

weights = curr_layer.weights;

future_grad = 0;
for i=1:length(future_layers)
    future_grad = future_grad + future_layers{i}.grad;
end

% gradient of the input
% grad_y = sum(future_grad,2);
grad_y = future_grad;
grad = grad_y * weights;

% gradient of the weights
grad_W = input' * grad_y;

if 1
    grad_W_raw = weights .* grad_W' - weights * (weights * grad_W);
else
    grad_W_raw = (diag(weights) - weights' * weights) * grad_W;
    grad_W_raw = grad_W_raw';
end

end
