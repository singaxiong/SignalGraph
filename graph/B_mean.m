function grad = B_mean(prev_layer, future_layers)
T = size(prev_layer{1}.a,2);

grad = 0;
for i=1:length(future_layers)
    % grad = grad + repmat(sum(future_layers{i}.grad,2), 1, T)/T;
    grad = grad + repmat(future_layers{i}.grad, 1, T)/T;
end

end
