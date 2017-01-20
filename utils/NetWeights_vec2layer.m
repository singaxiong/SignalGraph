function layer = NetWeights_vec2layer(vec, layer, packGrad)

WeightNames = WeightNameList('tunable');

% store weights into the vec
offset = 0;
for i=1:length(layer)
    if isfield(layer{i}, 'update') && layer{i}.update
        for j=1:length(WeightNames)
            if isfield(layer{i}, WeightNames{j})
                if isfield(layer{i}, 'mask') && strcmpi(WeightNames{j}, 'W')
                    para_idx = find(layer{i}.mask~=0);
                else
                    para_idx = 1:numel(layer{i}.(WeightNames{j}));
                end
                nPara = numel(para_idx);
                if packGrad
                    layer{i}.(['grad_' WeightNames{j}])(para_idx) = vec(offset+1:offset+nPara);
                else
                    layer{i}.(WeightNames{j})(para_idx) = vec(offset+1:offset+nPara);
                end
                
%                 [M,N] = size(layer{i}.(WeightNames{j}));
%                 nPara = M*N;
%                 if packGrad
%                     layer{i}.(['grad_' WeightNames{j}]) = reshape(vec(offset+1:offset+nPara), M,N);
%                 else
%                     layer{i}.(WeightNames{j}) = reshape(vec(offset+1:offset+nPara), M, N);
%                 end
                offset = offset + nPara;
            end
        end
    end
end