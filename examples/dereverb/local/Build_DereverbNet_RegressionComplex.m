% Build and initialize the computational graph for regression based speech
% enhancement/dereverberation
%
function [layer, para] = Build_DereverbNet_RegressionComplex(Data_tr, para)
para.output = 'tmp';

layer = genNetworkDereverb_RegressionComplex(para.topology);

para.preprocessing{1} = {};                     % optional preprocessing for each data stream
para.preprocessing{2} = {};
para.cost_func.layer_idx = length(layer);       % specify which layers are cost function layers
para.cost_func.layer_weight = [1];              % set the weights of each cost function layer
para = ParseOptions2(para);

idx = ReturnLayerIdxByName(layer, 'mu_law');
layer{idx(1)+1}.W = eye(layer{idx(1)+1}.dim(1)) * 3;
layer{idx(1)+1}.b = zeros(layer{idx(1)+1}.dim(1),1);

end