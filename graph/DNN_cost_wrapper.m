% convert the trainable parameters in the graph into a vector such that we
% can call standard optimization packages to optimize the network
% parameters in batch mode.
%
function [cost, grad] = DNN_cost_wrapper(W, layer, data, para, mode)

% retrieve the weights from W and assign it to the correct layers
layer = NetWeights_vec2layer(W, layer, 0);

% compute the forward and backward passes and get the gradients
[cost_func, layer] = DNN_Cost10(layer, data, para, mode);
cost = cost_func.cost;

% retrieve the gradient from layer and store it into vector format
[grad] = NetWeights_layer2vec(layer, 1, para.useGPU);

cost = gather(cost);
grad = gather(grad);

end
