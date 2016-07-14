function grad = B_cmn(future_layers)
future_grad = GetFutureGrad(future_layers, {});

[D, T, N] = size(future_grad);
if N==1
    grad = CMN(future_grad')';
else
    [mask, variableLength] = getValidFrameMask(future_layers{1});
    future_grad2 = ExtractVariableLengthTrajectory(future_grad, mask);
    for i=1:N
        grad{i} = CMN(future_grad2{i}')';
    end
    grad = cell2mat_gpu(grad);
    if variableLength
        grad = PadGradientVariableLength(grad, mask);
    end
end

end
