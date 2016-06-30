function output = F_tanh(input)
[D,M,N] = size(input);
output = tanh(input);

if N>1
    [mask, variableLength] = CheckTrajectoryLength(input);
    if variableLength; output = PadShortTrajectory(output, mask, -1e10); end
end
end
