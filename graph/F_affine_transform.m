function [output, validFrameMask] = F_affine_transform(input_layers, curr_layer)
if length(input_layers)==1
    input_layer = input_layers{1};
    W = curr_layer.W;
    b = curr_layer.b;
elseif length(input_layers)==2
    W = input_layers{1}.a;
    b = zeros(size(W,1),1);
    input_layer = input_layers{2};
end
input = input_layer.a;

[D,M,N] = size(input);
if N==1
    output = bsxfun(@plus, W * input, b);
    validFrameMask = [];
else
    [validFrameMask, variableLength] = getValidFrameMask(input_layer);
    %if variableLength; input = PadShortTrajectory(input, validFrameMask, 0); end
    input2 = reshape(input, D,M*N);
    output2 = bsxfun(@plus, W * input2, b);
    output = reshape(output2, size(output2,1), M,N);
end
end
