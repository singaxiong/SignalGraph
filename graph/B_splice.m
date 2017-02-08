function grad = B_splice(future_layer, curr_layer)
future_grad = GetFutureGrad(future_layer, curr_layer);
[dim,nFr,nSeg] = size(future_grad);
context = curr_layer.context;
dim = dim/context;
half_ctx = (context-1)/2;

if context==1
    grad = future_grad;
    return;
end

if nSeg>1
    [mask, variableLength] = getValidFrameMask(future_layer{1});
    if variableLength
        future_grad = PadShortTrajectory(future_grad, mask, 0);
    end
else
    variableLength = 0;
end
precision = class(gather(future_grad(1)));
if strcmpi(class(future_grad), 'gpuArray')
    grad = gpuArray.zeros(dim,nFr,nSeg, precision);
else
    grad = zeros(dim, nFr,nSeg, precision);
end
if 0
    for i=-half_ctx:half_ctx
        curr_future_grad = future_grad( (i+half_ctx)*dim+1 : (i+half_ctx+1)*dim, :);
        if i<0
            grad(:,1:end+i) = grad(:,1:end+i) + curr_future_grad(:, -i+1:end);
            grad(:,1) = grad(:,1) + sum(curr_future_grad(:, 1:-i),2);
        elseif i>0
            grad(:,i+1:end) = grad(:,i+1:end) + curr_future_grad(:, 1:end-i);
            grad(:,end) = grad(:,end) + sum(curr_future_grad(:, end-i+1:end),2);
        else
            grad = grad + curr_future_grad;
        end
    end
else
    if variableLength
        future_grad2 = ExtractVariableLengthTrajectory(future_grad, mask);
        for i=1:nSeg
            nFrSeg = size(future_grad2{i},2);
            grad(:,1:nFrSeg,i) = B_splice_single_sentence(grad(:,1:nFrSeg,i), future_grad(:,1:nFrSeg,i), context);
        end
    else
        grad = B_splice_single_sentence(grad, future_grad, context);
    end
end

end

