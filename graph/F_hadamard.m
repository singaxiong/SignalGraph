function [output, validFrameMask] = F_hadamard(input_layers)
input = input_layers{1}.a;
input2 = input_layers{2}.a;

output = input .* input2;

[D,M,N] = size(input);
if N==1
    validFrameMask = [];
else
    validFrameMask = getValidFrameMask(input_layers{1});
end

end
