function [gcc, maskGCC] = F_comp_gcc(input_layer, curr_layer)
input = input_layer.a;
[nCh,T,N] = size(input);
frame_len = curr_layer.frame_len;
frame_shift = curr_layer.frame_shift;
overlap = 1 - frame_shift/frame_len;
nFr = enframe_decide_frame_number(T, frame_len, frame_shift, 1-overlap/2);   % get the maximum number of frames of the sentences. It is
        % the same computation as in my_enframe.m
useGPU = IsInGPU(input);
precision = class(gather(input(1)));

gcc_dim = curr_layer.dim(1) / (nCh*(nCh-1)/2);
gcc_bin_range = (gcc_dim-1)/2;

if N==1     % if we have only one sentence in a minibatch
    [gcc]=getCorrelationVector_fast2(input, frame_len, overlap, useGPU);
    gcc = gcc(frame_len/2-gcc_bin_range:frame_len/2+gcc_bin_range,:,:);
    if nCh>2
        gcc = permute(gcc, [1,3,2]);
        [d1,d2,d3] = size(gcc);
        gcc = reshape(gcc, d1*d2, d3);
    end
    maskGCC = [];
else    % if we have multiple sentences in a minibatch
    [mask, variableLength] = getValidFrameMask(input_layer);
    input2 = PadShortTrajectory(input, mask, 0);
    
    % baseline
    if nCh>2    % loop over the sentences directly if too many channels
        if useGPU
            gcc = gpuArray.zeros(gcc_dim, nFr, nCh*(nCh-1)/2,N);
        else
            gcc = zeros(gcc_dim, nFr, nCh*(nCh-1)/2,N);
        end
        % add or remove samples to avoid partial samples to make enframe
        % faster
        nSampleRequired = nFr * frame_shift + frame_len-frame_shift;
        if nSampleRequired > T
            input2(:,nSampleRequired,:) = 0;
        end
        for i=1:N
            tmpGCC = getCorrelationVector_fast2(input2(:,:,i), frame_len, overlap, useGPU);
            gcc(:,:,:,i) = tmpGCC(frame_len/2-gcc_bin_range:frame_len/2+gcc_bin_range,:,:);
        end
        if nCh>2
            gcc = permute(gcc, [1,3,2,4]);
            [d1,d2,d3,d4] = size(gcc);
            gcc = reshape(gcc, d1*d2, d3,d4);
        else
            gcc = squeeze(gcc);
        end
        
    else        % put input into a big wavfile and call GCC function only once to save time.
        % just concatenate the sentences including the invalid samples
        % first discard the extra samples that will be discarded by enframe
        nSampleRequired = nFr * frame_shift + frame_len-frame_shift;
        if T<nSampleRequired
            input2(:,nSampleRequired,:) = 0;
        elseif T> nSampleRequired
            input2(:,nSampleRequired+1:end,:) = [];
        end
        % second, we need to append some zeros such that the first samples
        % of the sentences are guaranteed to be the first sample of a
        % frame. 
        residual = mod(frame_len/frame_shift,1);
        if residual>0
            nSampleToAppend = (1-residual)*frame_shift; % need to make the residual equal to the frame_shift
            input2(:,end+nSampleToAppend,:) = 0;
        end
        nFrActual = nFr + ceil(frame_len/frame_shift)-1;
        
        % third, reshape the tensor into a matrix
        input3 = reshape(input2, nCh, size(input2,2)*N);
        
        [gcc]=getCorrelationVector_fast2(input3, frame_len, overlap, useGPU);
        gcc = gcc(frame_len/2-gcc_bin_range:frame_len/2+gcc_bin_range,:,:);
        if size(gcc,2)<nFrActual*N
            gcc(:,nFrActual*N,:) = 0;
        end
        
        if nCh>2
            gcc = permute(gcc, [1,3,2]);
            [d1,d2,d3] = size(gcc);
            gcc = reshape(gcc, d1*d2, d3);
        end
        
        gcc = reshape(gcc, size(gcc,1), nFrActual, N);
        gcc(:,nFr+1:end,:) = [];    % remove extra frames
%     else    % concatenate the valid samples of the sentences without some buffer
        % to be implemented
    end
    
    % now build a mask for gcc
    nSampleChannel = gather(sum(mask==0));
    if useGPU
        maskGCC = gpuArray.zeros(nFr, N, precision);
    else
        maskGCC = zeros(nFr, N, precision);
    end
    for i=1:N
        nFrChannel = enframe_decide_frame_number(nSampleChannel(i), frame_len, frame_shift, 1-overlap/2);
        maskGCC(nFrChannel+1:end,i) = 1;
    end
    %gcc = PadShortTrajectory(gcc, maskGCC, -1e10);
end

end
