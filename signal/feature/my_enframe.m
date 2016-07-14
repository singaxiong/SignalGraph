function x_store = my_enframe(x, frame_size, frame_shift, useLastPartialFrame)
if nargin<4
    useLastPartialFrame = 0;    % whether to use the last few samples that are not as long as a frame
    % if set to >0, it represents the minimum percentage of a frame
    % required to generate the last frame
end

[nSample, nCh] = size(x);

% directly create an array of desired format. This will make the call to
% enframe_core faster
if nSample < 2^16
    frame_idx = (uint16(1) : uint16(nSample))';
elseif nSample<2^32
    frame_idx = (uint32(1) : uint32(nSample))';
else
    frame_idx = (1 : nSample)';
end

idx = enframe_core(frame_idx,frame_size,frame_shift, useLastPartialFrame);  % get the index of samples in each frame

if isempty(idx)
    x_store = [];
else
    if useLastPartialFrame>0 && idx(end,end)==0          % if there is 0 in the index, the last index must be zero.
        idx2 = max(1,idx);
        x_store = x(idx2,:);            % index all channels simultaneously for speed
        x_store(idx==0,:) = 0;          % sometimes, the last frames are padded with 0
    else
        x_store = x(idx,:);
    end
    x_store = reshape(x_store, frame_size, size(idx,2), nCh);
    x_store = permute(x_store, [1 3 2]);
end
end
