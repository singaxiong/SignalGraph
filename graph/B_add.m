function grad = B_add(prev_layers, future_layers)

future_grad = GetFutureGrad(future_layers, []);

for i=1:length(prev_layers)
    grad{i} = future_grad;
end

end
