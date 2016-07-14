function [output, validFrameMask] = F_affine_transform(input_layer, transform, bias)
input = input_layer.a;
[D,M,N] = size(input);
if N==1
    output = bsxfun(@plus, transform * input, bias);
    validFrameMask = [];
else
    [validFrameMask, variableLength] = getValidFrameMask(input_layer);
    %if variableLength; input = PadShortTrajectory(input, validFrameMask, 0); end
    input2 = reshape(input, D,M*N);
    output2 = bsxfun(@plus, transform * input2, bias);
    output = reshape(output2, size(output2,1), M,N);
end
end
