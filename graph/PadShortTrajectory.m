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
if strcmpi(padnumber, 'last')   % pad the last frame
    for i=1:N
        idx = find(mask(:,i)==1);
        if ~isempty(idx)
            output(:,idx,i) = repmat(input(:,idx(1)-1,i), 1, length(idx));
        end
    end
elseif padnumber==0
    idx = find(reshape(mask, 1, T*N)==1);
    if ~isempty(idx)
        output = reshape(output, D,T*N);
        output(:,idx) = padnumber;
        output = reshape(output, D,T,N);
    end
else
    tmp = squeeze(output(1,:,:));
    output(1,:,:) = tmp .* (1-mask) + padnumber*mask;
end

end
