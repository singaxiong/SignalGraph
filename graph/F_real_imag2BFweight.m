

function [output,validFrameMask] = F_real_imag2BFweight(input_layer, freq_bin, online)
% assume input is an array of time delay of C microphone channels. 
% freq_bin is an array of center frequencies of N FFT bins. 
input = input_layer.a;
[D, T, nSent] = size(input);
N = length(freq_bin);
nCh = D/N/2;
j = sqrt(-1);

% input_mean = mean(input,2);
% realpart = reshape(input_mean(1:N*nCh), N, nCh);
% imagpart = reshape(input_mean(N*nCh+1:end), N, nCh);

if online == 0
    if nSent>1
        [validFrameMask, variableLength] = getValidFrameMask(input_layer);
        if variableLength
            input2 = ExtractVariableLengthTrajectory(input,validFrameMask);    % remove the invalid frames
            precision = class(gather(input(1)));

            if IsInGPU(input)
                meanInput = gpuArray.zeros(D, 1, nSent, precision);
            else
                meanInput = zeros(D, 1, nSent, precision);
            end
            for i=1:nSent
                meanInput(:,1,i) = mean(input2{i},2);
            end
            input = meanInput;
        else
            input = mean(input,2);
        end
    else
        input = mean(input,2);
        validFrameMask = [];
    end
end

realpart = input(1:N*nCh,:,:);
imagpart = input(N*nCh+1:end,:,:);

output = realpart + j*imagpart;
end
