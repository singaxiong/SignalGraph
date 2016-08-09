% If a trajectory is shorter than others, as specified by the mask, we set
% its first dimension a defined number. There are two usage of this
% function:
%   1. pad the output trajectory of a layer with a big negative number, e.g. -1e10 to notice
%   other layers the trajectory is shorter than others in the minibatch.
%   this is often applied to hidden activations of CNN and LSTM.
%   2. pad the input trajectory of a layer with 0, so the computation will
%   carry out for all trajectories as if they have the same length, but do
%   not affect the results.
%
function output = PadShortTrajectory(input, mask, padnumber)
[~, N] = size(mask);
[D,T,N] = size(input);

output = input;
if ~sum(mask(:))  % all have the same length
    return;
end
if strcmpi(padnumber, 'last')   % pad the last frame
    % find the index of the last valide frame in all sequences
%     delta = mask(2:end,:) - mask(1:end-1,:);
%     [max_delta, max_idx] = max(delta);
%     max_idx(max_delta==0) = T;
    max_idx = gather(GetLastValidFrameIndex(mask));
    % pad the last valid frame to the invalide frames
    if 0    % 1 direct implementation
        for i=1:N
            output(:,max_idx(i)+1:end,i) = repmat(input(:,max_idx(i),i), 1, T-max_idx(i));
        end
    else    % 2 index based implementation
        fr_idx = reshape(1:T*N, T,N); %repmat( (1:T)', 1, N );
        for i=1:N
            fr_idx(max_idx(i)+1:end,i) = fr_idx(max_idx(i),i);
        end
        output = input(:,fr_idx);
        output = reshape(output,D,T,N);
    end
    
    % older implementation
%     for i=1:N
%         idx = gather(find(mask(:,i)==1,1));
%         if ~isempty(idx)
%             output(:,idx:end,i) = repmat(input(:,idx-1,i), 1, T-idx+1);
%             idx2(i) = idx;
%         end
%     end

elseif padnumber==0

    % simple implementation
    output(:,mask==1) = padnumber;
    
    % more complicated implementation
%     idx = find(reshape(mask, 1, T*N)==1);
%     if ~isempty(idx)
%         output = reshape(output, D,T*N);
%         output(:,idx) = padnumber;
%         output = reshape(output, D,T,N);
%     end
else
    tmp = squeeze(output(1,:,:));
    output(1,:,:) = tmp .* (1-mask) + padnumber*mask;
end

end
