function grad = B_cmn(future_layers)
future_grad = GetFutureGrad(future_layers, {});

[D, T, N] = size(future_grad);
if N==1
    grad = CMN(future_grad')';
else
    [mask, variableLength] = getValidFrameMask(future_layers{1});
    if variableLength
        future_grad2 = ExtractVariableLengthTrajectory(future_grad, mask);
        for i=1:N
            grad{i} = CMN(future_grad2{i}')';
        end
        grad = cell2mat_gpu(grad);
        
        grad = PadGradientVariableLength(grad, mask);
    else
        future_grad2 = reshape(permute(future_grad, [1 3 2]), D*N,T);
        grad = CMN(future_grad2')';
        grad = permute(reshape(grad, D, N, T), [1 3 2]);
    end
end

end
