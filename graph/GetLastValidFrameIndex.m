% given a mask of valid frames in sentences, return the an array of index of last
% valid frames in each sentence. 
%
function last_idx = GetLastValidFrameIndex(mask)
T = size(mask,1);
delta = mask(2:end,:) - mask(1:end-1,:);
[max_delta, last_idx] = max(delta);
last_idx(max_delta==0) = T;

end
