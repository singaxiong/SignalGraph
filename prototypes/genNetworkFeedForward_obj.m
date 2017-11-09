function layer = genNetworkFeedForward_obj(inputDim, hiddenLayerSize, outputDim, costFn, LastActivation4MSE)
if nargin<5
    LastActivation4MSE = 'linear';
end
inputDim = double(inputDim);

layer{1} = InputNode(1,1);
layer{end}.dim = [1 1]*inputDim;

for i=1:length(hiddenLayerSize)
    layer{end+1} = AffineNode(length(layer)+1);
    if i==1
        layer{end}.dim = [hiddenLayerSize(i) inputDim];
    else
        layer{end}.dim = hiddenLayerSize(i:-1:i-1);
    end
    
    layer{end+1} = SigmoidNode(length(layer)+1);
    layer{end}.dim = [1 1]*hiddenLayerSize(i);
end

layer{end+1} = AffineNode(length(layer)+1);
if length(hiddenLayerSize)==0
    layer{end}.dim = [outputDim inputDim];
else
    layer{end}.dim = [outputDim hiddenLayerSize(end)];
end

if strcmpi(costFn, 'CrossEntropy')
    layer{end+1} = SoftmaxNode(length(layer)+1);
    layer{end}.dim = [1 1]*outputDim;
end

layer{end+1} = InputNode(length(layer)+1,2);
layer{end}.dim = [1 1]*inputDim;

if strcmpi(costFn, 'CrossEntropy')
    layer{end+1} = CrossEntropyNode(length(layer)+1);
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
    layer{end+1} = MeanSquareErrorNode(length(layer)+1);
end
layer{end}.prev = [-2 -1];
layer{end}.dim = [1 outputDim];
layer = FinishLayer_obj(layer);
end

