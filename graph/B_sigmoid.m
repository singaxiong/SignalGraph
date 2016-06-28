function [grad]= B_sigmoid(future_layers, output)
future_grad = GetFutureGrad(future_layers, []);
grad = future_grad .* output .* (1-output);
[D,M,N] = size(future_grad);
if N>1
    [mask, variableLength] = CheckTrajectoryLength(future_grad);
    if variableLength; grad = PadShortTrajectory(grad, mask, -1e10); end
end

end