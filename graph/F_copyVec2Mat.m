% put a vector in selected positions of a matrix. 
%
function [output] = F_copyVec2Mat(input_layer, curr_layer)
input = input_layer.a;

[D,T,N] = size(input);
D1 = curr_layer.targetDims(1);
D2 = curr_layer.targetDims(2);
index2copy = curr_layer.index2copy;

if IsInGPU(input(1))
    output = gpuArray.zeros(D1, D2, T, N);
    tmp = gpuArray.zeros(D1,D2);
else
    output = zeros(D1, D2, T, N);
    tmp = zeros(D1,D2);
end

for t=1:T
    for n = 1:N
        tmp(index2copy) = input(:,t,n);
        output(:,:,t,n) = tmp;
    end
end

end
