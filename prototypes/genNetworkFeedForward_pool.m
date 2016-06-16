% poolType = 'mean' or 'max'
% poolLayer = integer layer number after which we do the pooling
function layer = genNetworkFeedForward_pool(inputDim, hiddenLayerSize, outputDim, costFn, poolType, poolAfterNlayer, LastActivation4MSE)
if nargin<7
    LastActivation4MSE = 'linear';
end

layer = genNetworkFeedForward_v2(inputDim, hiddenLayerSize, outputDim, costFn, LastActivation4MSE);

poolLayer.name = poolType;
poolLayer.dim = [1 1]*layer{poolAfterNlayer}.dim(1);
poolLayer.prev = layer{poolAfterNlayer}.prev;
layer = [layer(1:poolAfterNlayer) poolLayer layer(poolAfterNlayer+1:end)];

% automatically derive the list of layers that the output of the current layer goes.
for i=1:length(layer); layer{i}.next = []; end
for i=length(layer):-1:1
    if isfield(layer{i}, 'prev')
        for j=1:length(layer{i}.prev)
            layer{i+layer{i}.prev(j)}.next(end+1) = -layer{i}.prev(j);
        end
    end
end

end

