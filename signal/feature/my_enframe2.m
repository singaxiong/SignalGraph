function x_store = my_enframe2(x, frame_size, frame_shift, useLastPartialFrame)
if nargin<4
    useLastPartialFrame = 0;    % whether to use the last few samples that are not as long as a frame
    % if set to >0, it represents the minimum percentage of a frame
    % required to generate the last frame
end

nCh = size(x,2);

    idx = my_enframe((1:size(x,1))',frame_size,frame_shift);
    x_store = x(idx,:);
    x_store = reshape(x_store, frame_size, size(idx,3), nCh);
    x_store = permute(x_store, [1 3 2]);
end