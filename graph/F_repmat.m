% repeat a matrix 
%
function [output] = F_repmat(input_layer, curr_layer)
input = input_layer.a;

sourceDims = curr_layer.sourceDims;
targetDims = curr_layer.targetDims;

if length(sourceDims)==1
    output = repmat(input, targetDims(1),targetDims(2));
else
    % to be implemented
end

end