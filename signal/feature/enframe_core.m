%% enframe_core can enframe single channel data in CPU memory. 
% But it is better to use it to enframe the sample indexes for easier
% handling with GPU memory variables and faster processing of multi-channel
% data. 
function x_store = enframe_core(x, frame_size, frame_shift, useLastPartialFrame)
nFr = enframe_decide_frame_number(size(x,1), frame_size, frame_shift, useLastPartialFrame);

overlap = frame_size - frame_shift;
if license('test', 'Signal_Toolbox')
    x_store = buffer(x(overlap+1:end),frame_size,overlap, x(1:overlap));       % buffer will append 0 to the data to form integer number of frames
else
    x_store = enframe(x, frame_size, frame_shift)';
end
if size(x_store,2)>nFr
    x_store(:,nFr+1:end) = [];     % we may remove the last frames
end
end