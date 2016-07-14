
function [output,validFrameMask] = F_cmn(input_layer)
input = input_layer.a;
[D,T,N] = size(input);

if N==1
    output = CMN(input')';
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
            output(:,1:size(input2{i},2),i) = CMN(input2{i}')';
        end
    else
        input2 = reshape(permute(input, [1 3 2]), D*N,T);
        output = CMN(input2')';
        output = permute(reshape(output, D, N, T), [1 3 2]);
    end
end
end
