function [output, validFrameMask] = F_hadamard(input_layers)
input = input_layers{1}.a;
mask = input_layers{2}.a;

output = input .* mask;

[D,M,N] = size(input);
if N==1
    validFrameMask = [];
else
    validFrameMask = getValidFrameMask(input_layer);
end

end
