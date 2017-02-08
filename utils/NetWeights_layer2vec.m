% pack the weights or gradients of a network into a vector

function [vec] = NetWeights_layer2vec(layer, packGrad, useGPU)

WeightNames = WeightNameList('tunable');

% first get the total number of parameters
dim = 0;
for i=1:length(layer)
    for j=1:length(WeightNames)
        if isfield(layer{i}, WeightNames{j}) && layer{i}.update
            dim = dim + numel(layer{i}.(WeightNames{j}));
        end
    end
end

% allocate the memory
if useGPU
    vec = gpuArray.zeros(dim,1);
else
    vec = zeros(dim,1);
end
% store weights into the vec
offset = 0;
for i=1:length(layer)
    for j=1:length(WeightNames)
        if isfield(layer{i}, WeightNames{j}) && layer{i}.update
            if isfield(layer{i}, 'mask') && strcmpi(WeightNames{j}, 'W')
                para_idx = find(layer{i}.mask~=0);
            else
                para_idx = 1:numel(layer{i}.(WeightNames{j}));
            end
            nPara = numel(para_idx);
            
            if packGrad
                vec(offset+1:offset+nPara) = layer{i}.(['grad_' WeightNames{j}])(para_idx);
            else
                vec(offset+1:offset+nPara) = layer{i}.(WeightNames{j})(para_idx);
            end
            offset = offset + nPara;
        end
    end
end

end
