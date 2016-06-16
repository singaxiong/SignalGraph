% Synchronize multiple streams to have the same frame rate and number of
% frames.
function data_sync = SynchronizeStreamsFrameRate(data, frame_rate, isTensor)
for j=1:length(data)
    [n1(j),n2(j),n3(j)] = size(data{j});
    if isTensor(j) == 1 && n3(j)>1
        data{j} = reshape(data{j}, n1(j)*n2(j), n3(j));    % convert 3D tensor to matrix
    end        
end
if sum(n3==0)>0
    for j=1:length(data)
        data_sync{j} = [];
    end
    return;
end

if sum(abs( frame_rate - frame_rate(1) )) >0  % if frame rates are different
    maxFramerate = max(frame_rate);
    
    % find out the maximum number of frames after synchronization
    for j=1:length(data)
        if frame_rate(j) > 0
            nFrv(j) = size(data{j},2) * maxFramerate / frame_rate(j);
        end
    end
    nFrMax = max(nFrv);
    
    % we will align all features to the maxFramerate
    for i=1:length(data)
        if frame_rate(i)==0
            data{i} = repmat(data{i}, 1, nFrMax);
        elseif frame_rate(i) < maxFramerate
            % use interpolation to generate the missing frames
            data{i} = alignFeature_framerate(data{i}, frame_rate(i), maxFramerate, nFrMax);
        end
    end
end

% remove extra frames
for i=1:length(data)
    nFr(i) = size(data{i},2);
end
nFrMin = min(nFr);
for i=1:length(data)
    data_sync{i} = data{i}(:,1:nFrMin);     % all streams take the minimum number of frames
end

if sum(isTensor)    % convert matrix to tensor
    for j=1:length(data)
        if isTensor(j)
            if n3(j)>1
                data_sync{j} = reshape(data_sync{j}, n1(j), n2(j), size(data_sync{j},2));    % convert 3D tensor to matrix
            else
                if sum(n3>1)>0  % if some stream is 3D tensor, we also need to convert the current stream to tensor. 
                    data_sync{j} = reshape(data_sync{j}, n1(j), 1, max(n3));
                end
            end
        else
            data_sync{j} = reshape(data_sync{j}, n1(j), 1, size(data_sync{j},2));   % make the stream similar to Tensor
        end
    end
end

end
