function para = RemoveIOStream(para, streamIdxToBeRemoved)

nStreamRemove = length(streamIdxToBeRemoved);

para.nStream = para.nStream - nStreamRemove;

ArrayConfig = GetArrayConfig();

for i=1:length(ArrayConfig)
    if isfield(para, ArrayConfig{i})
        para.(ArrayConfig{i})(streamIdxToBeRemoved) = [];
    end
end

end