function nFr = enframe_decide_frame_number(nSample, frame_size, frame_shift, useLastPartialFrame)
if nargin<3
    useLastPartialFrame = 0;
end

nFr_raw = (nSample-frame_size)/frame_shift+1;
if useLastPartialFrame>0 && mod(nFr_raw,1)>useLastPartialFrame && mod(nFr_raw,1) < 1    % decide the number of frames
    nFr = ceil(nFr_raw);
else
    nFr = floor(nFr_raw);
end

end