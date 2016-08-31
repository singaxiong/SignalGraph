% reshape the first N dimsions of the data
% sourceDims: an array [D1 D2 D3,...] that specifies the dimensions to be
% involved in reshaping.
% targetDims: an array [M1 M2 M3,...] that specifies the dimensions after
% reshaping. prod(sourceDims) need to be eqal to prod(targetDims).
%
function [output] = F_reshape(input_layer, curr_layer)
input = input_layer.a;

sourceDims = curr_layer.sourceDims;
targetDims = curr_layer.targetDims;

[D(1), D(2), D(3), D(4)] = size(input);

reshapeSize = [targetDims(:)' D(length(sourceDims)+1:end)];

output = reshape(input, reshapeSize);

end