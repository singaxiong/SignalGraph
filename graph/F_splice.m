function [output,validFrameMask] = F_splice(input_layer, context)
input = input_layer.a;
[D, T, N] = size(input);
halfC = (context-1)/2;

validFrameMask = [];
if context==1
    output = input;
elseif N==1
    output = ExpandContext_v2(input, -halfC : halfC);
else
    [validFrameMask, variableLength] = getValidFrameMask(input_layer);
    if variableLength
        input = PadShortTrajectory(input, validFrameMask, 'last');
    end
    output = ExpandContext_v2(input, -halfC : halfC);
end


% [validFrameMask] = getValidFrameMask(input_layer);
% input2 = Tensor2MatWithBuffer(input, context, 1);
% output2 = ExpandContext_v2(input2, -halfC : halfC);
% output2 = reshape(output2, size(output2,1), T+context, N);
% output = output2(:,halfC+1:halfC+T,:);

if 0    % baseline implementation by looping over sentences
    output3 = gpuArray.zeros(D*context, T, N);
    for i=1:N
        output3(:,:,i) = ExpandContext_v2(input(:,:,i), -halfC : halfC);
    end
end
end