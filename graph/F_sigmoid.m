function output = F_sigmoid(input)
[D,M,N] = size(input);
output = sigmoid(input);

if N>1
    [mask, variableLength] = CheckTrajectoryLength(input);
    if variableLength; output = PadShortTrajectory(output, mask, -1e10); end
end
end
