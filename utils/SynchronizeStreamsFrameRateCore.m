% synchronize two streams' frame rates and number of frames
% each stream is assume to be a 3D tensor D_i x T_i x N. Note that the
% third dimension is assume to be the same for both stremas. 
function data_sync = SynchronizeStreamsFrameRateCore(data, frame_rate)

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
            [d1,d2,d3] = size(data{i});
            if d3>1
                dataTmp = reshape(permute(data{i}, [1 3 2]), d1*d3, d2);
                dataTmp2 = alignFeature_framerate(dataTmp, frame_rate(i), maxFramerate, nFrMax);
                data{i} = permute(reshape(dataTmp2, d1,d3,size(dataTmp2,2)), [1 3 2]);
            else
                data{i} = alignFeature_framerate(data{i}, frame_rate(i), maxFramerate, nFrMax);
            end
        end
    end
end

% remove extra frames
for i=1:length(data)
    nFr(i) = size(data{i},2);
end
nFrMin = min(nFr);
for i=1:length(data)
    data_sync{i} = data{i}(:,1:nFrMin,:);     % all streams take the minimum number of frames
end

end