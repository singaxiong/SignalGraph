% Build and initialize the computational graph for regression based speech
% enhancement/dereverberation
%
function [layer, para] = Build_EnhanceNet_Gaussian(Data_tr, para, stage)
para.output = 'tmp';

layer = genNetworkDereverb_Gaussian(para.topology, stage);     % generate the network graph
para.preprocessing{1} = {};                     % optional preprocessing for each data stream
para.preprocessing{2} = {};
para.cost_func.layer_idx = length(layer);       % specify which layers are cost function layers
para.cost_func.layer_weight = [1];              % set the weights of each cost function layer
para.IO.inputFeature = [1 1];
para.IO.isTensor = [1 1];
para = ParseOptions2(para);

% generating the scaling factor for the input, as we will need to use a
% small constant in the logarithm. We need to make sure that the power of
% speech are larger than this constant most of the time.
scale = 1e4;        % we hard code the scale to be a constant so that all network will use the same number
scale = scale/2^16; % note that we are using int16 to store waveform samples, so need to scale down
if para.topology.useWav
    layer = InitWavScaleLayer(layer, scale);
end

if stage>1 && isfield(para.topology, 'initModel')
    modelfiles = findFiles(['nnet/' para.topology.initModel], 'mat');
    modelfiles = sort_nnet_by_dev(modelfiles);
    fprintf('Use initial model %s\n', modelfiles{1});
    dnnInit = load(modelfiles{1});
    
    % find the shared, mean, and var subnets
    [sharedLayers, meanLayers, varLayers] = findSubnets(layer);
    if stage == 3.3     % initialize with regression model, use the initial model only for mean prediction
        [sharedLayersInit, meanLayersInit] = findSubnetsRegression(dnnInit.layer);        
    elseif stage == 3.2     % initialize with regression model. share the LSTM between mean and variance prediction
        [sharedLayersInit, meanLayersInit] = findSubnetsRegression(dnnInit.layer);
    else            % initialize with stage 1 or 2 model
        [sharedLayersInit, meanLayersInit, varLayersInit] = findSubnets(dnnInit.layer);        
    end
    layer = CopyNetWeightsByLayerIdx(dnnInit.layer, [sharedLayersInit meanLayersInit], layer, [sharedLayers meanLayers]);
    if floor(stage) ==3 && stage~=3.1 && stage~=3.2 && stage~=3.3   % for stage 2, we don't initilize var subnet
        layer = CopyNetWeightsByLayerIdx(dnnInit.layer, varLayersInit, layer, varLayers);
    end
else
    if strcmpi(para.topology.RegressionNetType, 'DNN')    % if use DNN, splice the frames
        idx = ReturnLayerIdxByName(layer, 'splice');
    else
        idx = ReturnLayerIdxByName(layer, 'delta'); % if use LSTM, use dynamic features
    end
    fft_net_length = idx(1);
    fft_net = layer(1:fft_net_length);
    paraTmp = para;
    paraTmp.out_layer_idx = fft_net_length;
    paraTmp.IO = RemoveIOStream(paraTmp.IO, [2]);
    paraTmp.IO.nStream = 1;
    paraTmp.IO.inputFeature = paraTmp.IO.DynamicDistortion.inputFeature(1);
    if isfield(paraTmp.IO, 'fileReader'); paraTmp.IO = rmfield(paraTmp.IO, 'fileReader'); end
    paraTmp.IO.fileReader(1)= paraTmp.IO.DynamicDistortion.fileReader(1);
    fprintf('Generate global MVN weights for mask subnet - %s\n', datestr(now));
    [layer{fft_net_length+1}.W, layer{fft_net_length+1}.b] = computeGlobalCMVN(Data_tr(1), 100, paraTmp, fft_net);
    VerifyPreprocessingTree(layer(1:fft_net_length+1), Data_tr(1), paraTmp, 100);
end
end

function [sharedLayers, meanLayers, varLayers] = findSubnets(layer)

delta_idx = ReturnLayerIdxByName(layer, 'delta');
input_affine_idx = delta_idx(1)+1;
tmp = length(layer) + layer{end}.prev;
mean_idx = tmp(1);
var_idx = tmp(2);
var_start = mean_idx+1;
sharedOut_idx = layer{var_start}.prev + var_start;
sharedLayers = (input_affine_idx:sharedOut_idx);
meanLayers = (sharedOut_idx+1:var_start-1);
varLayers = (var_start:var_idx);

end

function [sharedLayers, meanLayers] = findSubnetsRegression(layer)

delta_idx = ReturnLayerIdxByName(layer, 'delta');
input_affine_idx = delta_idx(1)+1;
tmp = length(layer) + layer{end}.prev;
mean_idx = delta_idx(2);

sharedOut_idx = mean_idx-2;
sharedLayers = (input_affine_idx:sharedOut_idx);
meanLayers = (sharedOut_idx+1:mean_idx);

end