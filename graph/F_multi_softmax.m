function output = F_multi_softmax(input_layer, TaskVocabSizes)
input = input_layer.a;
[D,T,N] = size(input);
    
if N>1; [mask, variableLength] = CheckTrajectoryLength(input); end
    
if length(TaskVocabSizes)==1    % this is for when you have the same task, e.g. skip gram
    if D>TaskVocabSizes
        input2 = reshape(input, TaskVocabSizes, D/TaskVocabSizes, T, N);
        fakeLayer.a = input2;
        output = F_softmax(fakeLayer);
        output = reshape(output, D, T, N);
    else        % this is normal softmax
        output = F_softmax(input_layer);
    end
    
else    % this is for when you have different tasks
    % to be implemented
end

if N>1 && variableLength; output = PadShortTrajectory(output, mask, -1e10); end

end
