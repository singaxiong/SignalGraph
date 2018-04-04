% generate a network with linear structure using descriptive config

function layer = genNetworkLinearStructure(config)

for i=1:length(config)
    currConfig = config{i};
    
    
end


inputDim = double(inputDim);

inputStreamIdx = 1;
layer{1} = InputNode(inputStreamIdx,inputDim);

for i=1:length(hiddenLayerSize)
    layer{end+1} = AffineNode(hiddenLayerSize(i));
    layer{end+1} = SigmoidNode(hiddenLayerSize(i));
end
layer{end+1} = AffineNode(outputDim);

if strcmpi(costFn, 'CrossEntropy')
    layer{end+1} = SoftmaxNode(outputDim);
else    % there are a few options for MSE cost function
    switch LastActivation4MSE
        case 'linear'
            % do nothing
        case 'tanh'
            layer{end+1} = TanhNode(outputDim);
        case 'sigmoid'
            layer{end+1} = SigmoidNode(outputDim);
        case 'relu'
            layer{end+1} = ReluNode(outputDim);
        otherwise
            fprintf('Error: unknown activation for last hidden layer, skipped!\n');
    end
end
layer = ConnectNodesLinear(layer);

inputStreamIdx = inputStreamIdx +1;
if strcmpi(costFn, 'CrossEntropy')
    layer{end+1} = InputNode(inputStreamIdx, 1);
    layer{end+1} = CrossEntropyNode();
else
    layer{end+1} = InputNode(inputStreamIdx, outputDim);
    layer{end+1} = MeanSquareErrorNode(1);
end
layer{end}.prev = [-2 -1];
layer{end}.dim = [1 outputDim];
layer = FinishLayer_obj(layer);
end

