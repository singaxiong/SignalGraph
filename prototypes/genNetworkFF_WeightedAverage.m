% generate network prototype that use feedforward (FF) networks for
% predicting feature representation and frame weights. Then the feature
% representation are weighted by the frame weights and summed together to
% produce a single feature vector for each input sequence.

function layer = genNetworkFF_WeightedAverage(inputDim, hiddenLayerSizeFeature, hiddenLayerSizeWeight, hiddenLayerSizeAfterAverage, useSigmoidBeforeAverage, average_before_activiation, outputDim, costFn)
inputDim = double(inputDim);

layer1 = genNetworkFeedForward_v2(inputDim, [hiddenLayerSizeFeature hiddenLayerSizeAfterAverage], outputDim, costFn);

layer2 = genNetworkFeedForward_v2(inputDim, hiddenLayerSizeWeight, 1, costFn);

if useSigmoidBeforeAverage
    if average_before_activiation
        weight_net_start_index = 2*length(hiddenLayerSizeFeature) + 1;
    else
        weight_net_start_index = 2*length(hiddenLayerSizeFeature) + 2;
    end
else
    weight_net_start_index = 2*length(hiddenLayerSizeFeature) + 1;
    layer1(weight_net_start_index) = [];
end

feature_net = layer1(1:weight_net_start_index-1);
post_net = layer1(weight_net_start_index:end);
weight_net = layer2(2:end-3);
weight_net{1}.prev = 1-weight_net_start_index;

weighted_average_node{1}.name = 'weighted_average';
weighted_average_node{1}.prev = [-1 -1-length(weight_net)];

layer = [feature_net weight_net weighted_average_node post_net];


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

