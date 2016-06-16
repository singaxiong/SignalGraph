function layer = genNetworkIvectorMeanPool(inputDim, hiddenSize, outputDim, bottleneckLayerIdx, useMeanPool, useSigmoidBeforeMeanPool)
% inputDim = 440;
% hiddenSize = [2048 2048 400 2048];
% outputDim = 3000;
% bottleneckLayerIdx = 3;

layer{1}.name = 'Input';        % this is an input layer
layer{end}.inputIdx = 1;    % specifies the index of GCC in Visible_tr
layer{end}.dim = [1 1]*inputDim;             % [input dim; output dim];

for i=1:bottleneckLayerIdx
    layer{end+1}.name = 'Affine';
    layer{end}.prev = -1;   % this is the index of GCC features in Visible_tr; It is the offset to be added to current layer index
    layer{end}.W = []; % to be initialized randomly or by pretraining
    layer{end}.b = [];
    if i==1
        layer{end}.dim = [hiddenSize(i) inputDim];
    else
        layer{end}.dim = [hiddenSize(i) hiddenSize(i-1)];
    end
    layer{end}.update = 1;
    
    layer{end+1}.name = 'sigmoid';
    layer{end}.prev = -1;
    layer{end}.dim = [1 1]*hiddenSize(i);
end

if useMeanPool
    if useSigmoidBeforeMeanPool==0
        layer = layer(1:end-1);
    end
    layer{end+1}.name = 'mean';
    layer{end}.prev = -1;
end

for i=bottleneckLayerIdx+1:length(hiddenSize)
    layer{end+1}.name = 'Affine';
    layer{end}.prev = -1;   % this is the index of GCC features in Visible_tr; It is the offset to be added to current layer index
    layer{end}.W = []; % to be initialized randomly or by pretraining
    layer{end}.b = [];
    layer{end}.dim = [hiddenSize(i) hiddenSize(i-1)];
    layer{end}.update = 1;
    
    layer{end+1}.name = 'sigmoid';
    layer{end}.prev = -1;
    layer{end}.dim = [1 1]*hiddenSize(i);
end

layer{end+1}.name = 'Affine';
layer{end}.prev = -1;   % this is the index of GCC features in Visible_tr; It is the offset to be added to current layer index
layer{end}.W = []; % to be initialized randomly or by pretraining
layer{end}.b = [];
layer{end}.dim = [outputDim hiddenSize(end)];
layer{end}.update = 1;

layer{end+1}.name = 'Softmax';
layer{end}.prev = -1;
layer{end}.dim = [1 1]*outputDim;

layer{end+1}.name = 'Input';
layer{end}.inputIdx = 2;
layer{end}.dim = [1 1]*outputDim;

layer{end+1}.name = 'Cross_Entropy';
layer{end}.prev = [-2 -1];
layer{end}.dim = [1 outputDim];

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
