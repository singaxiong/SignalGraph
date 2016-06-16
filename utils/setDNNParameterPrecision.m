function layer = setDNNParameterPrecision(layer, singlePrecision)
if singlePrecision
    for i=1:length(layer)
        if isfield(layer{i}, 'W')
            layer{i}.W = single(layer{i}.W);
        end
        if isfield(layer{i}, 'b')
            layer{i}.b = single(layer{i}.b);
        end
    end
else
    for i=1:length(layer)
        if isfield(layer{i}, 'W')
            layer{i}.W = double(layer{i}.W);
        end
        if isfield(layer{i}, 'b')
            layer{i}.b = double(layer{i}.b);
        end
    end
end
end
