function grad = B_repmat(future_layers, curr_layer)
future_grad = GetFutureGrad(future_layers, curr_layer);

sourceDims = curr_layer.sourceDims;
% targetDims = curr_layer.targetDims;

if length(sourceDims)==1
    grad = sum(future_grad,2);
else
    % to be implemented
end

end