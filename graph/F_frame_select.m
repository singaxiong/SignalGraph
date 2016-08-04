% select frames from the input stream
% 
function [output, validFrameMask] = F_frame_select(input_layer, curr_layer)
input = input_layer.a;
[D,T,N] = size(input);

words = ExtractWordsFromString_v2(curr_layer.frameSelect);
selectionType = words{1};

switch selectionType
    case 'last'     % only select the last N frames
        if length(words)>1; nFrameSelect = str2num(words{2}); else nFrameSelect = 1; end
        if N>1; [mask, variableLength] = GetValidFrameMask(input_layer); else variableLength = 0; end
        if variableLength
            last_idx = gather(GetLastValidFrameIndex(mask));    % index of last valid frame in all sentences
            if sum(last_idx<nFrameSelect); fprintf('Error: some sentences in the minibatch is shorter than the number of selected length\n'); return; end
            if IsInGPU(input); output = gpuArray.zeros(D,nFrameSelect,N); else output = zeros(D,nFrameSelect,N); end
            for i=1:N
                output(:,:,i) = input{i}(:,(last_idx(i)-nFrameSelect+1):last_idx(i));
            end
        else
            output = input(:,end-nFrameSelect+1:end,:);
        end
        validFrameMask = zeros(nFrameSelect,N);

    case 'first'    % only select the first frames
        % to be implemented
end


end
