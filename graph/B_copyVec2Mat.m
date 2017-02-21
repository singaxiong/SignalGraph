% put a vector in selected positions of a matrix. 
%
function [grad] = B_copyVec2Mat(input_layer, curr_layer, future_layers)
input = input_layer.a;

[D,T,N] = size(input);
D1 = curr_layer.targetDims(1);
D2 = curr_layer.targetDims(2);
index2copy = curr_layer.index2copy;

future_grad = GetFutureGrad(future_layers, curr_layer);

if IsInGPU(input(1))
    grad = gpuArray.zeros(size(input));
else
    output = zeros(size(input));
end

for t=1:T
    for n = 1:N
        tmp = future_grad(:,:,t,n);
        grad(:,t,n) = tmp(index2copy);
    end
end

end