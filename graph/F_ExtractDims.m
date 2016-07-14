function [output] = F_ExtractDims(input_layer, idx)
input = input_layer.a;
output = input(idx,:,:);

end