function [grad, grad_W, grad_b, validFrameMask]= B_affine_transform(input_layers, curr_layer, future_layers, skip_grad)
if length(input_layers)==1
    input = input_layers{1}.a;
    W = curr_layer.W;
    b = curr_layer.b;
elseif length(input_layers)==2
    W = input_layers{1}.a;
    b = zeros(size(W,1),1);
    input = input_layers{2}.a;
end
validFrameMask = [];

if curr_layer.update==0
    grad_W = [];
    grad_b = [];
end

future_grad = GetFutureGrad(future_layers, curr_layer);

[n1,n2,n3] = size(future_grad);
if n3>1     % reshape the matrix to 2D
    [validFrameMask, variableLength] = getValidFrameMask(input_layers{length(input_layers)});
    if variableLength
        future_grad = PadShortTrajectory(future_grad, validFrameMask, 0); 
%         input = PadShortTrajectory(input, validFrameMask, 0);
%         future_grad = future_grad(:,validFrameMask==0);
%         input = input(:,validFrameMask==0);
    end
    future_grad = reshape(future_grad, n1,n2*n3);
    input = reshape(input, size(input,1), n2*n3);
end

if curr_layer.update
    if issparse(input)
        visible_nonzero_idx = find(sum(abs(input),2)>0);
        visible_nonzero = full(input(visible_nonzero_idx,:));
        grad_W = sparse(size(W,1),size(W,2));
        grad_W(:,visible_nonzero_idx) = future_grad * visible_nonzero';
    else
        grad_W = conj(future_grad * input');
    end
    if isfield(curr_layer, 'mask')
        grad_W = grad_W .* curr_layer.mask;
    end
    if isfield(curr_layer, 'updateBias') && ~curr_layer.updateBias  % sometimes, we don't use bias
        grad_b = future_grad(:,1)*0;
    else
        grad_b = sum(future_grad,2);
    end
end
if isfield(curr_layer, 'skip_grad')
    skip_grad = skip_grad + curr_layer.skip_grad;
end
if skip_grad==0 
    grad = W' * future_grad;
    if n3>1
        grad = reshape(grad, size(grad,1), n2, n3);
    end
else
    grad = [];
end

end
