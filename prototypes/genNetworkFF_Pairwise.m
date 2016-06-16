% generate network prototype that use feedforward (FF) networks for
% predicting feature representation and frame weights. Then the feature
% representation are weighted by the frame weights and summed together to
% produce a single feature vector for each input sequence.

function [layer, WeightTyingSet] = genNetworkFF_Pairwise(inputDim, hiddenLayerSize, distanceBeforeActivation, distance_type, costFn)
inputDim = double(inputDim);
outputDim = 1;

layer1 = genNetworkFeedForward_v2(inputDim, hiddenLayerSize, outputDim, costFn);

if distanceBeforeActivation
    feature_net_end_index = 2*length(hiddenLayerSize);
else
    feature_net_end_index = 2*length(hiddenLayerSize) + 1;
end

feature_net = layer1(1:feature_net_end_index);
post_net = layer1(feature_net_end_index+1:end);
feature_net2 = feature_net;
feature_net2{1}.inputIdx = 2;

dist_node{1}.name = distance_type;
dist_node{1}.prev = [-1 -1-length(feature_net)];
dist_node{1}.dim = [1 1];

LT_node{1}.name = 'affine_transform';
LT_node{1}.prev = -1;
LT_node{1}.W = [];
LT_node{1}.b = [];
LT_node{1}.dim = [1 1];
LT_node{1}.update =1;

layer = [feature_net feature_net2 dist_node LT_node];
WeightTyingSet = {};
for i=1:length(feature_net)
    if isfield(layer{i}, 'W')
        WeightTyingSet{end+1} = [i, i+length(feature_net)];
    end
end

if strcmpi(costFn, 'logistic')
    layer{end+1}.name = 'Input';
    layer{end}.inputIdx = 3;
    layer{end}.dim = [1 1]*outputDim;
    
    layer{end+1}.name = 'logistic';
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

