% generate network prototype that use feedforward (FF) networks for
% predicting feature representation and frame weights. Then the feature
% representation are weighted by the frame weights and summed together to
% produce a single feature vector for each input sequence.

function [layer, WeightTyingSet] = genNetworkMTL_DML_CE(inputDim, hiddenLayerSize, outputDim, distanceBeforeActivation, dist_type, cost_function)


[layer, WeightTyingSet] = genNetworkFF_Pairwise2(inputDim, hiddenLayerSize, distanceBeforeActivation, dist_type, cost_function);

layer2 = genNetworkFeedForward_v2(hiddenLayerSize(end), [], outputDim, 'cross_entropy');

layer2(1) = [];
layer2{1}.prev = length(hiddenLayerSize)*2 -  length(layer) - 1;
layer2{end-1}.inputIdx = 4;

layer3 = layer2;
layer3{1}.prev = length(hiddenLayerSize)*4 -  length(layer) -1 - length(layer2);
layer3{end-1}.inputIdx = 5;

layer = [layer layer2 layer3];

layer = FinishLayer(layer);

WeightTyingSet{end+1} = [-7 -3]+length(layer);

end

