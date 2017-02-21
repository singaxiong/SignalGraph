% Build and initialize the computational graph for regression based speech
% enhancement/dereverberation
%
function [layer, para] = Build_DereverbNet_Masking(Data_tr, para)
para.output = 'tmp';

layer = genNetworkDereverb_Masking(para.topology);     % generate the network graph
para.preprocessing{1} = {};                     % optional preprocessing for each data stream
para.preprocessing{2} = {};
para.cost_func.layer_idx = length(layer);       % specify which layers are cost function layers
para.cost_func.layer_weight = [1];              % set the weights of each cost function layer
para = ParseOptions2(para);

% generating the scaling factor for the input, as we will need to use a
% small constant in the logarithm. We need to make sure that the power of
% speech are larger than this constant most of the time. 
scale = 1e4;        % we hard code the scale to be a constant so that all network will use the same number
scale = scale/2^16; % note that we are using int16 to store waveform samples, so need to scale down
if para.topology.useWav
    layer = InitWavScaleLayer(layer, scale);
end

if strcmpi(para.topology.RegressionNetType, 'DNN')    % if use DNN, splice the frames
    idx = ReturnLayerIdxByName(layer, 'splice');
else
    idx = ReturnLayerIdxByName(layer, 'delta'); % if use LSTM, use dynamic features
end
fft_net_length = idx(1);
fft_net = layer(1:fft_net_length);
paraTmp = para;
paraTmp.out_layer_idx = fft_net_length;
fprintf('Generate global MVN weights for mask subnet - %s\n', datestr(now));
[layer{fft_net_length+1}.W, layer{fft_net_length+1}.b] = computeGlobalCMVN(Data_tr, 100, paraTmp, fft_net);
VerifyPreprocessingTree(layer(1:fft_net_length+1), Data_tr, paraTmp, 100);

% set weight of static, velocity, and accelration features in the MSE cost
% function. 
delta_idx = ReturnLayerIdxByName(layer, 'delta');
weight_idx = delta_idx+1;
for i=1:length(weight_idx)
    if isfield(layer{weight_idx(i)}, 'W') && numel(layer{weight_idx(i)}.W) == prod(layer{weight_idx(i)}.dim)
        continue;
    end
    layer{weight_idx(i)}.W = diag([ones(para.topology.nFreqBin,1)*para.topology.MSECostWeightSDA(1); ...
        ones(para.topology.nFreqBin,1)*para.topology.MSECostWeightSDA(2); ...
        ones(para.topology.nFreqBin,1)*para.topology.MSECostWeightSDA(3)]);
    layer{weight_idx(i)}.b = zeros(para.topology.nFreqBin*3,1);
end

end