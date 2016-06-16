function layer = genNetworkFeedForward(inputDim, nHiddenLayer, hiddenLayerSize, outputDim, costFn)
inputDim = double(inputDim);

layer{1}.name = 'Input';        % this is an input layer
layer{end}.inputIdx = 1;    % specifies the index of GCC in Visible_tr
layer{end}.dim = [1 1]*inputDim;             % [input dim; output dim];

for i=1:nHiddenLayer
    layer{end+1}.name = 'Affine';
    layer{end}.prev = -1;   % this is the index of GCC features in Visible_tr; It is the offset to be added to current layer index
    layer{end}.W = []; % to be initialized randomly or by pretraining
    layer{end}.b = [];
    if i==1
        layer{end}.dim = [hiddenLayerSize inputDim];
    else
        layer{end}.dim = [1 1]*hiddenLayerSize;
    end
    layer{end}.update = 1;
    
    layer{end+1}.name = 'sigmoid';
    layer{end}.prev = -1;
    layer{end}.dim = [1 1]*hiddenLayerSize;
end

layer{end+1}.name = 'Affine';
layer{end}.prev = -1;
layer{end}.W = []; % to be initialized randomly or by pretraining
layer{end}.b = [];
layer{end}.dim = [outputDim hiddenLayerSize];
layer{end}.update = 1;

if strcmpi(costFn, 'cross_entropy')
    layer{end+1}.name = 'softmax';
    layer{end}.prev = -1;
    layer{end}.dim = [1 1]*outputDim;
end

layer{end+1}.name = 'Target';
layer{end}.targetIdx = 1;
layer{end}.dim = [1 1]*outputDim;

if strcmpi(costFn, 'cross_entropy')
    layer{end+1}.name = 'cross_entropy';
    layer{end}.prev = [-2 -1];
    layer{end}.dim = [1 outputDim];
else
    layer{end+1}.name = 'MSE';
    layer{end}.prev = [-2 -1];
    layer{end}.dim = [1 outputDim];
    layer{end}.useMahaDist = 0;
end

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

