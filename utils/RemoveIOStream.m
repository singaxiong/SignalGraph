function para = RemoveIOStream(para, streamIdxToBeRemoved)

nStreamRemove = length(streamIdxToBeRemoved);

para.nStream = para.nStream - nStreamRemove;

ArrayConfig = GetArrayConfig();

for i=1:length(ArrayConfig)
    if isfield(para, ArrayConfig{i})
        para.(ArrayConfig{i})(streamIdxToBeRemoved) = [];
    end
end

for i=1:length(para.DataSyncSet)
    if length(para.DataSyncSet{i})>1
        for j=1:length(streamIdxToBeRemoved)
            idx = para.DataSyncSet{i}==streamIdxToBeRemoved(j);
            if sum(idx)
                para.DataSyncSet{i}(idx) = [];  % remove the stream from DataSyncSet if it is already removed
            end
        end
    end
    if length(para.DataSyncSet{i})<2    % if there is less than 2 streams, no need to sync
        para.DataSyncSet{i} = [];
    end
end

end