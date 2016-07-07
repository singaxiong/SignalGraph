function layer = NetWeights_vec2layer(vec, layer, packGrad)

WeightNames = WeightNameList('tunable');

% store weights into the vec
offset = 0;
for i=1:length(layer)
    if isfield(layer{i}, 'update') && layer{i}.update
        for j=1:length(WeightNames)
            if isfield(layer{i}, WeightNames{j})
                [M,N] = size(layer{i}.(WeightNames{j}));
                nPara = M*N;
                if packGrad
                    layer{i}.(['grad_' WeightNames{j}]) = reshape(vec(offset+1:offset+nPara), M,N);
                else
                    layer{i}.(WeightNames{j}) = reshape(vec(offset+1:offset+nPara), M, N);
                end
                offset = offset + nPara;
            end
        end
    end
end