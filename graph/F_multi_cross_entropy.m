function [cost,acc] = F_multi_cross_entropy(input_layers, CE_layer)
[nSeg, output, target] = prepareCostEvaluation(input_layers, CE_layer);

TaskVocabSizes = CE_layer.TaskVocabSizes;

if length(TaskVocabSizes)==1    % this is for when you have the same task, e.g. skip gram
    [D,T] = size(output);
    if D>TaskVocabSizes
%         output = reshape(output, TaskVocabSizes, D/TaskVocabSizes, T);
%         target = reshape(target, 1, D/TaskVocabSizes, T);
        output = reshape(output, TaskVocabSizes, D/TaskVocabSizes*T);
        target = reshape(target, 1, D/TaskVocabSizes*T);
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
    cost = -1/m*sum(log(output2));
   
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