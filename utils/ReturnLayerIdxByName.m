function layer_idx = ReturnLayerIdxByName(layer, layer_name)

layer_idx = [];
for i=1:length(layer)
    if strcmpi(layer{i}.name, layer_name)
        layer_idx(end+1) = i;
    end
end
end
