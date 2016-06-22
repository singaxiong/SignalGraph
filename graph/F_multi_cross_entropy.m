function [cost,acc] = F_multi_cross_entropy(input_layers, CE_layer)
[nSeg, output, target, scale] = prepareCostEvaluation(input_layers, CE_layer);
if ~isempty(scale); hasScale = 1; else hasScale = 0; end

TaskVocabSizes = CE_layer.TaskVocabSizes;

if length(TaskVocabSizes)==1    % this is for when you have the same task, e.g. skip gram
    [D,T] = size(output);
    nTasks = D/TaskVocabSizes;
    if nTasks>1
        output = reshape(output, TaskVocabSizes, nTasks*T);
        target = reshape(target, 1, nTasks*T);
    end
    m = size(output,2);
    [~, recogClass] = max(output);
    % [~, trueClass] = max(target);
    trueClass = target;
    acc = sum(recogClass==trueClass)/m;
    
    dim = size(output,1);
    offset = 0:dim:m*dim-1;
    idx = offset+trueClass;
    output2 = output(idx);
    if hasScale
        output3 = reshape(output2, nTasks, T);
        cost = -1/sum(scale)/nTasks*sum( log(output3) * scale');
    else
        cost = -1/m*sum(log(output2));
    end   
else    % this is for when you have different tasks
    % to be implemented
end

if 0
    figure(1);
    imagesc(output); hold on
    plot(target,'r'); hold off;
    title(cost)
%     pause
end

end