% Synchronize multiple streams to have the same frame rate and number of
% frames.
% isTensor specifies whether we should treat the data streams as matrix or
% tensor.
function data_sync = SynchronizeStreamsFrameRate2(data, frame_rate, isTensor)
for j=1:length(data)
    [D(j),T(j),N(j)] = size(data{j});
end

if sum(D==0) || sum(T==0) || sum(N==0)  % if any stream is empty, put all stream as empty
    for j=1:length(data)
        data_sync{j} = zeros(0,0,0);
    end
    return;
end

% If a stream has T>1 and N>1, we treat it as multiple trajectories. 
% If a stream has T>1 and N=1, we treat it as a single trajectory.
% If a stream has T=1 and N>1, we treat it as multiple trajectory of length 1.
% If a stream has T=1 and N=1, we treat it as a single trajectory of length 1. 

if N(1)==1 && N(2)==1           % both stream have only 1 trajectory
    data_sync = SynchronizeStreamsFrameRateCore(data, frame_rate);
elseif N(1)==1 || N(2)==1       % one and only one stream is a single stream
    stream_single = find(N==1);     % the index of the stream containing only one trajectory
    stream_multi = find(N>1);
    
    if T(1)>1 && T(2)>1  || ( T(stream_multi)>1 && T(stream_single)==1 && frame_rate(stream_single)>0 )     % if both stream has length>1, we take only 1 trajectory from both stream
        for i=1:length(data)
            data_sync{i} = data{i}(:,:,1);
        end
        data_sync = SynchronizeStreamsFrameRateCore(data_sync, frame_rate);
    elseif T(stream_multi) == 1   % if both streams have length 1, repeat the single trajectory stream to match the other
        % upsample single trajectory stream 
        dataTmp = repmat(data{stream_single}, 1, max(N));
        data_sync{stream_single} = reshape(dataTmp, D(stream_single), max(T), max(N));
        if T(stream_single) > 1     % upsample multi trajectory stream
            dataTmp = repmat(data{stream_multi}, 1, max(T));
            data_sync{stream_multi} = reshape(dataTmp, D(stream_multi), max(T), max(N));
        else
            data_sync{stream_multi} = data{stream_multi};
        end
    else    % if one stream has multiple longer than 1 trajectories, the other has one trajectory of length 1.
        if frame_rate(stream_single)==0   % if the single trajectory stream has frame rate 0, we assume that it is the label and should be replicated to match the other stream
            dataTmp = repmat(data{stream_single}, 1, max(T)*max(N));
            data_sync{stream_single} = reshape(dataTmp, D(stream_single), max(T), max(N));
            data_sync{stream_multi} = data{stream_multi};
        end
    end
else
    % if both streams have multiple trajectories, the first thing to do is to make the number of trajectories the same
    minN = min(N);
    for i=1:length(data)
        data_sync{i} = data{i}(:,:,1:minN);
    end
    
    if T(1)==1 && T(2)==1   % both stream are multiple trajectories of length 1.
        % do nothing
    elseif T(1)==1 || T(2)==1   % one and only one trajectory has length 1.
        stream_1frame = find(T==1);   
        data_sync{stream_1frame} = repmat(data_sync{stream_1frame}, 1, max(T), 1);
    else    % both streams are multiple trajectories of length > 1
        data_sync = SynchronizeStreamsFrameRateCore(data_sync, frame_rate);
    end
end

end
