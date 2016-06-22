function [grad] = B_multi_softmax_multi_cross_entropy(input_layers_of_cross_entropy, CE_layer)
[nSeg, output, target, scale, nFrOrig, mask] = prepareCostEvaluation(input_layers_of_cross_entropy, CE_layer);
if ~isempty(scale); hasScale = 1; else hasScale = 0; end
if ~isempty(mask); variableLength = 1; else variableLength = 0; end
D = size(output,1); 
nFr = size(output,2)/nSeg;

TaskVocabSizes = CE_layer.TaskVocabSizes;

if length(TaskVocabSizes)==1    % this is for when you have the same task, e.g. skip gram
    [D,T] = size(output);
    nTasks = D/TaskVocabSizes;
    if nTasks>1
        output = reshape(output, TaskVocabSizes, nTasks*T);
        target = reshape(target, 1, nTasks*T);
    end
    m = size(output,2);
    trueClass = target;
    dim = size(output,1);
    offset = 0:dim:m*dim-1;
    idx = offset+trueClass;
    
    % grad2 = output;
    output(idx) = output(idx) - 1;
    if hasScale
        grad = reshape(output, D,T)/sum(scale)/nTasks;
        grad = bsxfun(@times, grad, scale);
    else
        grad = output/m;
        if nTasks>1
            grad = reshape(grad,D,T);
        end
    end
end 
if variableLength
    if isfield(CE_layer, 'labelDelay') && CE_layer.labelDelay~=0
        % when there is label delay, the mask does not match the grad
        % anymore, we need to make a modified mask that fit the grad
        nFrActual = sum(mask==0);
        if CE_layer.labelDelay>0
            mask = mask(CE_layer.labelDelay+1:end,:);
        else
            mask = mask(1:end-CE_layer.labelDelay,:);
            for i=1:nSeg
                mask(nFrActual(i)-CE_layer.labelDelay+1:end,i) = 1;
            end
        end
    end
    grad = PadGradientVariableLength(grad, mask);
    
    if isfield(CE_layer, 'labelDelay') && CE_layer.labelDelay~=0
        grad = ShiftGradient(grad, CE_layer.labelDelay);
    end
    
    if isfield(CE_layer, 'costFrameSelection')
        [grad] = AssignCostGradFrame(grad, nFrOrig, nSeg, mask, CE_layer);
    end
else
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

end
