function layer = genNetworkFeedForward_v2(inputDim, hiddenLayerSize, outputDim, costFn, LastActivation4MSE)
if nargin<5
    LastActivation4MSE = 'linear';
end
inputDim = double(inputDim);

layer{1}.name = 'Input';        % this is an input layer
layer{end}.inputIdx = 1;    % specifies the index of GCC in Visible_tr
layer{end}.dim = [1 1]*inputDim;             % [input dim; output dim];

for i=1:length(hiddenLayerSize)
    layer{end+1}.name = 'Affine';
    layer{end}.prev = -1;   % this is the index of GCC features in Visible_tr; It is the offset to be added to current layer index
    layer{end}.W = []; % to be initialized randomly or by pretraining
    layer{end}.b = [];
    if i==1
        layer{end}.dim = [hiddenLayerSize(i) inputDim];
    else
        layer{end}.dim = hiddenLayerSize(i:-1:i-1);
    end
    layer{end}.update = 1;
    
    layer{end+1}.name = 'sigmoid';
    layer{end}.prev = -1;
    layer{end}.dim = [1 1]*hiddenLayerSize(i);
end

layer{end+1}.name = 'Affine';
layer{end}.prev = -1;
layer{end}.W = []; % to be initialized randomly or by pretraining
layer{end}.b = [];
if length(hiddenLayerSize)==0
    layer{end}.dim = [outputDim inputDim];
else
    layer{end}.dim = [outputDim hiddenLayerSize(end)];
end
layer{end}.update = 1;

if strcmpi(costFn, 'cross_entropy')
    layer{end+1}.name = 'softmax';
    layer{end}.prev = -1;
    layer{end}.dim = [1 1]*outputDim;
end

layer{end+1}.name = 'Input';
layer{end}.inputIdx = 2;
layer{end}.dim = [1 1]*outputDim;

if strcmpi(costFn, 'cross_entropy')
    layer{end+1}.name = 'cross_entropy';
    layer{end}.prev = [-2 -1];
    layer{end}.dim = [1 outputDim];
else
    hasActivation = 0;
    switch LastActivation4MSE
        case 'linear'
            % do nothing
        case 'tanh'
            lastActivation.name = 'tanh';
            hasActivation = 1;
        case 'sigmoid'
            lastActivation.name = 'sigmoid';
            hasActivation = 1;
        case 'relu'
            lastActivation.name = 'relu';
            hasActivation = 1;
        otherwise
            fprintf('Error: unknown activation for last hidden layer, skipped!\n');
    end
    if hasActivation
        lastActivation.prev = -1;
        lastActivation.dim = [1 1]*layer{end-1}.dim(1);
        layer = [layer(1:end-1) lastActivation layer(end)];
    end
    layer{end+1}.name = 'MSE';
    layer{end}.prev = [-2 -1];
    layer{end}.dim = [1 outputDim];
    layer{end}.useMahaDist = 0;
end

layer = FinishLayer(layer);
end

