
function [output,validFrameMask] = F_beamforming(input_layers)
% assume input is a 3D tensor of complex values of size NxTxC, where D is the number of frequency bins, T is
% the number of frames, and C is the number of microphone channels.
% Assume weight is a 2D matrix of complex values of size NxC.
[input, weight, input_layer] =  prepareBeamforming(input_layers);
[N,C,T,nSent] = size(input);
if size(weight,3)==T
    online = 1;
else
    online = 0;
end

validFrameMask = [];
if nSent>1
    [validFrameMask] = getValidFrameMask(input_layer);
end

if online
    output = input .* conj(weight);
    output = squeeze(sum(output,2));
else
    if strcmpi(class(input),'gpuArray')==0
        output2 = 0;
        for i=1:C
            curr_channel = squeeze(input(:,i,:));
            output2 = output2 + bsxfun(@times, curr_channel, conj(weight(:,i)));
        end
        output = output2;
    else    % faster implementation
        output = arrayfun(@times, input, conj(weight));
        output = squeeze(sum(output,2));
    end
end

end
