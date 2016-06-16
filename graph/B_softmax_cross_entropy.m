function [grad] = B_softmax_cross_entropy(input_layers_of_cross_entropy, CE_layer)
[nSeg, output, target, scale, nFrOrig, mask] = prepareCostEvaluation(input_layers_of_cross_entropy, CE_layer);
m = size(output,2);
D = size(output,1); 
nFr = size(output,2)/nSeg;

if 0
    grad = -1/m * (target - output);
else
    %[~, trueClass] = max(target);
    trueClass = target;
    dim = size(output,1);
    offset = 0:dim:m*dim-1;
    idx = offset+trueClass;
    
    % grad2 = output;
    output(idx) = output(idx) - 1;
    grad = output/m;
end 
if nSeg>1     % we reshape the gradient to match the output size
    grad = reshape(grad, size(grad,1), nFr, nSeg);
end

if isfield(CE_layer, 'labelDelay') && CE_layer.labelDelay~=0
    grad = ShiftGradient(grad, CE_layer.labelDelay);
end
if isfield(CE_layer, 'costFrameSelection')
    [grad] = AssignCostGradFrame(grad, nFrOrig, nSeg, mask, CE_layer);
end
end
