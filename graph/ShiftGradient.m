function grad = ShiftGradient(grad, labelDelay)
if labelDelay==0
    return;
end
[D, nFr, nSeg] = size(grad);

if labelDelay > 0
    gradTmp = grad;
    precision = class(gather(grad(1,1,1)));
    if strcmpi(class(grad), 'gpuArray')
        grad = gpuArray.zeros(D,nFr+labelDelay, nSeg, precision);
    else
        grad = zeros(D,nFr+labelDelay, nSeg, precision);
    end
    grad(:,labelDelay+1:end,:) = gradTmp;
else
    grad(:,end+labelDelay,:) = 0;
end
end

