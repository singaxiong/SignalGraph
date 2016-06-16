function [data_sync] = SynchronizeDataStreams(data, para)

DataSyncSet = para.IO.DataSyncSet;
if length(DataSyncSet) == 0
    data_sync = data;
    return;
end

for di=1:length(DataSyncSet)
    curr_sync_set = DataSyncSet{di};
    if length(curr_sync_set) == 0; continue; end
    
    curr_frame_rate = para.IO.frame_rate(curr_sync_set);
    curr_data = data(curr_sync_set);
    
    if sum(abs( curr_frame_rate - curr_frame_rate(1) )) >0  % if frame rates are different
        maxFramerate = max(curr_frame_rate);
        
        % find out the maximum number of frames after synchronization
        for j=1:length(curr_data)
            if curr_frame_rate(j) > 0
                nFrv(j) = size(curr_data{j},2) * maxFramerate / curr_frame_rate(j);
            end
        end
        nFrMax = max(nFrv);        
        
        % we will align all features to the maxFramerate
        for i=1:length(curr_data)
            if curr_frame_rate(i)==0
                curr_data{i} = repmat(curr_data{i}, 1, nFrMax);
            elseif curr_frame_rate(i) < maxFramerate
                % use interpolation to generate the missing frames
                curr_data{i} = alignFeature_framerate(curr_data{i}, curr_frame_rate(i), maxFramerate, nFrMax);
            end
        end
    end
    
    % remove extra frames
    for i=1:length(curr_data)
        nFr{di}(i) = size(curr_data{i},2);
    end
    nFrMin(di) = min(nFr{di});
    for i=1:length(curr_data)
        data_sync{curr_sync_set(i)} = curr_data{i}(:,1:nFrMin(di));
    end
end

all_sync_set = cell2mat(DataSyncSet);
for i=1:length(data)
    if sum(i==all_sync_set)==0
        data_sync{i} = data{i};
    end
end
end