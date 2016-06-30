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
else % when we have different tasks
    % to be implemented
end
grad = PostprocessCostEvaluation(grad, output, mask, nSeg, nFrOrig, CE_layer);

end
