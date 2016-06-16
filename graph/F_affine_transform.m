function output = F_affine_transform(input, transform, bias)
[D,M,N] = size(input);
if N==1
    output = bsxfun(@plus, transform * input, bias);
else
    [mask, variableLength] = CheckTrajectoryLength(input);
    if variableLength; input = PadShortTrajectory(input, mask, 0); end
    input2 = reshape(input, D,M*N);
    output2 = bsxfun(@plus, transform * input2, bias);
    output = reshape(output2, size(output2,1), M,N);
    if variableLength; output = PadShortTrajectory(output, mask, -1e10); end
end
end
