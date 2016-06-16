function [grad, grad_W, grad_b]= B_affine_transform(prev_layer, curr_layer, future_layers, skip_grad)

transform = curr_layer.W;
grad = [];

if curr_layer.update==0
    grad_W = [];
    grad_b = [];
end

future_grad = GetFutureGrad(future_layers, curr_layer);
input = prev_layer{1}.a;

[n1,n2,n3] = size(future_grad);
if n3>1     % reshape the matrix to 2D
    [mask, variableLength] = CheckTrajectoryLength(future_grad);
    if variableLength; 
        future_grad = PadShortTrajectory(future_grad, mask, 0); 
        input = PadShortTrajectory(input, mask, 0);
    end
    future_grad = reshape(future_grad, n1,n2*n3);
    input = reshape(input, size(input,1), n2*n3);
end

if curr_layer.update
    if issparse(input)
        visible_nonzero_idx = find(sum(abs(input),2)>0);
        visible_nonzero = full(input(visible_nonzero_idx,:));
        grad_W = sparse(size(transform,1),size(transform,2));
        grad_W(:,visible_nonzero_idx) = future_grad * visible_nonzero';
    else
        grad_W = future_grad * input';
    end
    grad_b = sum(future_grad,2);
end
if skip_grad==0
    grad = transform' * future_grad;
end

if n3>1
    grad = reshape(grad, size(grad,1), n2, n3);
    if variableLength; grad = PadShortTrajectory(grad, mask, -1e10); end
end

end
