
function [output,validFrameMask] = F_absmax_norm(input_layer, curr_layer)
input = input_layer.a;
[D,T,N] = size(input);

if isfield(curr_layer, 'minmax')
    absmax = curr_layer.max;
else
    absmax = 1;
end

if N==1
    output = AbsMaxNorm(input,absmax);
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
            output(:,1:size(input2{i},2),i) = AbsMaxNorm(input2{i},absmax);
        end
    else
        for i=1:N
            output = AbsMaxNorm(input,absmax);
        end
    end
end
end
