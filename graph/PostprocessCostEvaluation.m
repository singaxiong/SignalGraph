% This function postprocess gradient to undo whatever we did in
% prepareCostEvaluation, such as labelDelay, costFrameSelection, etc. 
% Author: Xiong Xiao, Temasek Labs, NTU, Singapore. 
% Last modified: 28 Jun 2016
%
function grad = PostprocessCostEvaluation(grad, output, mask, nSeg, nFrOrig, CostLayer)
if ~isempty(mask) && sum(sum(mask))>0
    variableLength = 1; 
else variableLength = 0; end

if variableLength
    if isfield(CostLayer, 'costFrameSelection')
        [grad] = AssignCostGradFrame(grad, nFrOrig, nSeg, mask, CostLayer);
    end
    
    if isfield(CostLayer, 'labelDelay') && CostLayer.labelDelay~=0
        % when there is label delay, the mask does not match the grad
        % anymore, we need to make a modified mask that fit the grad
        nFrActual = sum(mask==0);
        if CostLayer.labelDelay>0
            mask = mask(CE_layer.labelDelay+1:end,:);
        else
            mask = mask(1:end-CE_layer.labelDelay,:);
            for i=1:nSeg
                mask(nFrActual(i)-CE_layer.labelDelay+1:end,i) = 1;
            end
        end
    end
    grad = PadGradientVariableLength(grad, mask);
    
    if isfield(CostLayer, 'labelDelay') && CostLayer.labelDelay~=0
        grad = ShiftGradient(grad, CE_layer.labelDelay);
    end
else
    nFr = size(output,2)/nSeg;
    if nSeg>1     % we reshape the gradient to match the output size
        grad = reshape(grad, size(grad,1), nFr, nSeg);
    end
    if isfield(CostLayer, 'costFrameSelection')
        [grad] = AssignCostGradFrame(grad, nFrOrig, nSeg, mask, CostLayer);
    end
    if isfield(CostLayer, 'labelDelay') && CostLayer.labelDelay~=0
        grad = ShiftGradient(grad, CE_layer.labelDelay);
    end
end

end