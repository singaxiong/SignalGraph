% shift the frames of the input stream

function [output, validFrameMask] = F_frame_shift(input_layer, curr_layer)
input = input_layer.a;
[D,T,N] = size(input);
if N>1; [mask, variableLength] = GetValidFrameMask(input_layer); else variableLength = 0; end
delay = curr_layer.delay;

if delay>0     % positive delay means that we delay the input
    output = input(:,1:end-delay, :);
    validFrameMask = mask(1:end-delay,:);
    if variableLength
        nFrActual = gather(sum(mask==0));   % we need to gather nFrActual as it will be used as index range and must not be in GPU memory
        for i=1:nSeg
            validFrameMask(nFrActual(i)-delay+1:end) = 1;
        end
    end
else                % negative delay means that we fast forward the input
    output = input(:,delay+1:end, :);
    validFrameMask = mask(delay+1:end,:);
end

end
