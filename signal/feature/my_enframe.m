function x_store = my_enframe(x, frame_size, frame_shift, useLastPartialFrame)
if nargin<4
    useLastPartialFrame = 0;    % whether to use the last few samples that are not as long as a frame
    % if set to >0, it represents the minimum percentage of a frame
    % required to generate the last frame
end

[nSample, nCh] = size(x);

idx = enframe_core((1:nSample)',frame_size,frame_shift, useLastPartialFrame);  % get the index of samples in each frame
if isempty(idx)
    x_store = [];
else
    if useLastPartialFrame>0
        idx2 = max(1,idx);
        x_store = x(idx2,:);      % index all channels simultaneously for speed
        x_store(idx==0,:) = 0;      % sometimes, the last frames are padded with 0
    else
        x_store = x(idx,:);
    end
    x_store = reshape(x_store, frame_size, size(idx,2), nCh);
    x_store = permute(x_store, [1 3 2]);
end
end

%% enframe_core can enframe single channel data in CPU memory. 
% But it is better to use it to enframe the sample indexes for easier
% handling with GPU memory variables and faster processing of multi-channel
% data. 
function x_store = enframe_core(x, frame_size, frame_shift, useLastPartialFrame)
overlap = frame_size - frame_shift;
N_block_raw = (size(x,1)-frame_size)/frame_shift+1;
if useLastPartialFrame>0 && mod(N_block_raw,1)>useLastPartialFrame && mod(N_block_raw,1) < 1    % decide the number of frames
    N_block = ceil(N_block_raw);
else
    N_block = floor(N_block_raw);
end

x_store = buffer(x(overlap+1:end),frame_size,overlap, x(1:overlap));       % buffer will append 0 to the data to form integer number of frames
if size(x_store,2)>N_block
    x_store(:,N_block+1:end) = [];     % we may remove the last frames
end
end