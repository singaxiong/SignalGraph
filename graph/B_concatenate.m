function grad = B_concatenate(prev_layer, curr_layer, future_layers)

future_grad = GetFutureGrad(future_layers, curr_layer);

for i=1:length(prev_layer)
    dim(i) = size(prev_layer{i}.a,1);
    grad{i} = future_grad( 1+ sum(dim(1:i-1)) : sum(dim), :);
end
end
