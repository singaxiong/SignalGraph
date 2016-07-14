
function [output,validFrameMask] = F_log(input_layer, const)
input = input_layer.a;
[D,T,N] = size(input);

if strcmpi(class(gather(input(1))), 'single')
    const = single(const);
end

if N==1
    output = log(input+const);
    validFrameMask = [];
else
    [validFrameMask, variableLength] = getValidFrameMask(input_layer);
    if variableLength
        input = PadShortTrajectory(input, validFrameMask, 0);
        if sum(input(:)+const<=0)
            fprintf('error: input to log is negative');
        end
        output = log(input+const);
    else
        output = log(input+const);
    end
end
end
