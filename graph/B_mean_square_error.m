function grad = B_mean_square_error(input_layers, useMahaDist, CostLayer)
[nSeg, output, target, scale, nFrOrig, mask] = prepareCostEvaluation(input_layers, CostLayer);
if ~isempty(mask); variableLength = 1; else variableLength = 0; end
m = size(output,2);
D = size(output,1); 
nFr = size(output,2)/nSeg;

diff = output - target;
if useMahaDist ==1  % in case 1, we use diagonal covariance matrix, which can be class-dependent
   grad = cost_scale * diff;
elseif useMahaDist ==2  % in case 2, we use full covariance matrix, which can only be global and not class-dependent
   grad = cost_scale * diff;
else
   grad = diff;
end
grad = grad / m;

if nSeg>1     % we are having multiple sequences in one minibatch
    grad = reshape(grad, D,nFr, nSeg);
end

if isfield(CostLayer, 'labelDelay') && CostLayer.labelDelay~=0
    grad = ShiftGradient(grad, CostLayer.labelDelay);
end
if isfield(CostLayer, 'costFrameSelection')
    switch CostLayer.costFrameSelection
        case 'last'

    end
end
end