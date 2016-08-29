% select frames from the input stream
% 
function grad = B_frame_select(input_layer, future_layers, curr_layer)
input = input_layer.a;
[D,T,N] = size(input);

future_grad = GetFutureGrad(future_layers, curr_layer);

words = ExtractWordsFromString_v2(curr_layer.frameSelect);
selectionType = words{1};
switch selectionType
    case 'last'
        if length(words)>1; nFrameSelect = str2num(words{2}); else nFrameSelect = 1; end
        if N>1; [mask, variableLength] = GetValidFrameMask(input_layer); else variableLength = 0; end
        
        precision = class(gather(future_grad(1,1,1)));
        if strcmpi(class(future_grad), 'gpuArray')
            grad = gpuArray.zeros(D,T, N, precision);
        else
            grad = zeros(D,T, N, precision);
        end
        if variableLength
            last_idx = gather(GetLastValidFrameIndex(mask));
            for i=1:nSeg
                grad(:,(last_idx(i)-nFrameSelect+1):last_idx(i)) = future_grad;
            end
        else
            grad(:,max(1,end-nFrameSelect+1):end,:) = future_grad;
        end
end

end
