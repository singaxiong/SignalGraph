function [output] = F_reshape(input_layer, curr_layer)
input = input_layer.a;
dim = curr_layer.dim;

[D,T,N] = size(input);

output = reshape(input, dim(1), D/dim(1), T);

output = permute(output, [1 3 2]);
end