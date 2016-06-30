function [grad]= B_tanh(future_layers, output)
future_grad = GetFutureGrad(future_layers, []);
grad = (1-output.^2) .* future_grad;

[D,M,N] = size(future_grad);
if N>1
    [mask, variableLength] = CheckTrajectoryLength(future_grad);
    if variableLength; grad = PadShortTrajectory(grad, mask, -1e10); end
end

end