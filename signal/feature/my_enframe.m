function x_store = my_enframe(x, frame_size, frame_shift, useLastPartialFrame)
if nargin<4
    useLastPartialFrame = 0;    % whether to use the last few samples that are not as long as a frame
    % if set to >0, it represents the minimum percentage of a frame
    % required to generate the last frame
end

if strcmpi(class(x(1)), 'gpuArray')
    useGPU = 1;
    x = gather(x);
else
    useGPU = 0;
end

N_block_raw = (size(x,1)-frame_size)/frame_shift+1;
N_block = floor(N_block_raw);
overlap = frame_size - frame_shift;
nCh = size(x,2);
needed_size = (N_block-1)*frame_shift + frame_size;

x_store = zeros(frame_size, nCh, N_block);
for ii=1:nCh
    x_store(:,ii,:) = buffer(x(overlap+1:needed_size,ii),frame_size,overlap);
end
x_store(:,:,1) = x(1:frame_size,:);
x_store(:,:,2) = x(frame_shift+1:frame_shift+frame_size,:);

if useLastPartialFrame && N_block < N_block_raw
    idx1 = N_block*frame_shift + 1;
    lastFrame = x(idx1:end,:);
    nSampleLastFrame = size(lastFrame,1);
    if nSampleLastFrame/frame_size > useLastPartialFrame
        x_store(1:nSampleLastFrame,:, end+1) = lastFrame;
    end
end
if useGPU == 1
    x_store = gpuArray(x_store);
end
end