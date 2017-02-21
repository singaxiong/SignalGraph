
function [output,validFrameMask] = F_minmax_norm(input_layer, curr_layer)
input = input_layer.a;
[D,T,N] = size(input);

if isfield(curr_layer, 'minmax')
    minmax = curr_layer.minmax;
else
    minmax = [-1 1];
end

if N==1
    output = MinMaxNorm(input,minmax(1), minmax(2));
    validFrameMask = [];
else
    [validFrameMask, variableLength] = getValidFrameMask(input_layer);
    if variableLength
        input2 = ExtractVariableLengthTrajectory(input, validFrameMask);
        precision = class(gather(input(1)));
        if IsInGPU(input)
            output = gpuArray.zeros(size(input), precision);
        else
            output = zeros(size(input), precision);
        end
        for i=1:N
            output(:,1:size(input2{i},2),i) = MinMaxNorm(input2{i},minmax(1), minmax(2));
        end
    else
        for i=1:N
            output = MinMaxNorm(input,minmax(1), minmax(2));
        end
    end
end
end
