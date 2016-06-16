function [grad] = B_multi_softmax_multi_cross_entropy(input_layers_of_cross_entropy, CE_layer)
[nSeg, output, target, scale, nFrOrig, mask] = prepareCostEvaluation(input_layers_of_cross_entropy, CE_layer);
D = size(output,1); 
nFr = size(output,2)/nSeg;

TaskVocabSizes = CE_layer.TaskVocabSizes;

if length(TaskVocabSizes)==1    % this is for when you have the same task, e.g. skip gram
    [D,T] = size(output);
    if D>TaskVocabSizes
        output = reshape(output, TaskVocabSizes, D/TaskVocabSizes*T);
        target = reshape(target, 1, D/TaskVocabSizes*T);
    end
    m = size(output,2);
    trueClass = target;
    dim = size(output,1);
    offset = 0:dim:m*dim-1;
    idx = offset+trueClass;
    
    % grad2 = output;
    output(idx) = output(idx) - 1;
    grad = output/m;
    
    if D>TaskVocabSizes
        grad = reshape(grad,D,T);
    end
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
