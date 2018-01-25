function grad = B_concatenate(prev_layer, curr_layer, future_layers)

future_grad = GetFutureGrad(future_layers, curr_layer);
[n1,n2,n3] = size(future_grad);

for i=1:length(prev_layer)
    dim(i) = size(prev_layer{i}.a,1);
    grad{i} = future_grad( 1+ sum(dim(1:i-1)) : sum(dim), :);
    if n3>1
        grad{i} = reshape(grad{i}, size(grad{i},1), n2,n3);
    end
end


end
