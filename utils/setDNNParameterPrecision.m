% set the weights precision to desired precision. Move weights to GPU if
% required. 
%
function layer = setDNNParameterPrecision(layer, singlePrecision, useGPU)

WeightNames = WeightNameList('all');

for i=1:length(layer)
    for j=1:length(WeightNames)
        if isfield(layer{i}, WeightNames{j})
            if singlePrecision
                layer{i}.(WeightNames{j}) = single(layer{i}.(WeightNames{j}));
            else
                layer{i}.(WeightNames{j}) = double(layer{i}.(WeightNames{j}));
            end
            if useGPU
                layer{i}.(WeightNames{j}) = gpuArray(layer{i}.(WeightNames{j}));
            end
        end
    end
end
end
