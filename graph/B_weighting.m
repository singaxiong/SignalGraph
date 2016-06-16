function [grad_X, grad_W, grad_b] = B_weighting(prev_layer, curr_layer, future_layers)

input = prev_layer{1}.a;
weight = curr_layer.W;
nWeight = length(weight);
D = size(input,1)/nWeight;

grad_X = 0; grad_W=0; grad_b=0;
future_grad = GetFutureGrad(future_layers, curr_layer);
if 0
    for i=1:length(future_layers)
        if curr_layer.update
            for j=1:size(input,2)
                curr_input = reshape(input(:,j), D, nWeight);
                grad_W = grad_W + curr_input' * future_layers{i}.grad(:,j);
            end
            if isfield(curr_layer, 'updateBias')== 0 || curr_layer.updateBias==1
                grad_b = grad_b + sum(future_layers{i}.grad,2);
            end
        end
        for j=1:size(input,2)
            grad_X = grad_X + future_layers{i}.grad(:,j)*weight';
        end
    end
else    
    if curr_layer.update
        for j=1:size(input,2)
            curr_input = reshape(input(:,j), D, nWeight);
            grad_W = grad_W + curr_input' * future_grad(:,j);
        end
        if isfield(curr_layer, 'updateBias')== 0 || curr_layer.updateBias==1
            grad_b = grad_b + sum(future_grad,2);
        end
    end
    for j=1:size(input,2)
        grad_X = grad_X + future_grad(:,j)*weight';
    end
end
grad_X = reshape(grad_X, prod(size(grad_X)), 1);

end
