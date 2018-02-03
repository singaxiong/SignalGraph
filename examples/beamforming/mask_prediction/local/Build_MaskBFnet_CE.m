% Build and initialize the computational graph for mask-based beamforming
% for speech recognition. 
% stage: 
%   1 - only initialize mask prediction subnet
%   2 - only initiliaze mask prediction and weight prediction subnets
%   3 - initialize also the acoustic model subnet
%
function [layer, para] = Build_MaskBFnet_CE(Data_tr, para, stage)
para.output = 'tmp';

layer = genNetworkMaskBF_CE(para.topology);     % generate the network graph
para.preprocessing{1} = {};                     % optional preprocessing for each data stream
para.preprocessing{2} = {};
if isfield(para.topology, 'MTL') && para.topology.MTL
    para.cost_func.layer_idx = [ReturnLayerIdxByName(layer, 'cross_entropy') length(layer)];       % specify which layers are cost function layers
    para.cost_func.layer_weight = [1 para.topology.MTL];              % set the weights of each cost function layer
    para.preprocessing{3} = {};
else
    para.cost_func.layer_idx = length(layer);       % specify which layers are cost function layers
    para.cost_func.layer_weight = [1];              % set the weights of each cost function layer    
end

% generating the scaling factor for the input, as we will need to use a
% small constant in the logarithm. We need to make sure that the power of
% speech are larger than this constant most of the time. 
scale = 1e4;        % we hard code the scale to be a constant so that all network will use the same number
scale = scale/2^16; % note that we are using int16 to store waveform samples, so need to scale down
layer = InitWavScaleLayer(layer, scale);

% copy the weights of the mask predicting subnet if provided

if isfield(para.topology, 'initialMask') && ~isempty(para.topology.initialMask) % copy BF DNN parameters and preprocessing from an initial Mask network
    tmp = load(para.topology.initialMask);
    para.preprocessing{1} = tmp.para.preprocessing{1};
    layerToCopy = ReturnLayerIdxByName(tmp.layer, 'affine');
    layerToCopy = [layerToCopy ReturnLayerIdxByName(tmp.layer, 'lstm')];
    layerToCopy = sort(layerToCopy);
    layer = CopyNetWeightsByLayerIdx(tmp.layer, layerToCopy(2:end), layer, layerToCopy(2:end) );   % skip the first affine transform (the scaling layer) that may have different sizes in initial model and CE model.
else    % otherwise, we need to generate the preprocessing
    if strcmpi(para.topology.MaskNetType, 'DNN')    % if use DNN, splice the frames
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
    
    % initialize weights randomly
    SpatialCovMask_idx = ReturnLayerIdxByName(layer, 'SpatialCovMask');
    layer(1:SpatialCovMask_idx) = initializeParametersDNN_tree(layer(1:SpatialCovMask_idx), ParseOptions2(para));
end

if stage<=1     % we only want to build the mask predicting subnet
    % remove the layers not belonging to mask subnet
    SpatialCovMask_idx = ReturnLayerIdxByName(layer, 'SpatialCovMask');
    layer = layer(1:SpatialCovMask_idx-1);
    % add a MSE cost function
    tmpLayer = genNetworkFeedForward_v2(layer{end}.dim(1), [], layer{end}.dim(1), 'mse');
    layer = [layer tmpLayer(end-1:end)];
    layer = FinishLayer(layer);
    para.cost_func.layer_idx = length(layer);
    return; 
end

if isfield(para.topology, 'initialBF') && ~isempty(para.topology.initialBF) % copy BF DNN parameters and preprocessing from an initial BF network
    % to be implemented
else    % otherwise, we need to generate the preprocessing
    switch para.topology.BfNetType
        case 'MVDR'
            % do nothing
        case {'DNN', 'LSTM'}
            % to be implemented
    end
end

if ~strcmpi(para.topology.BfNetType, 'MVDR')    % if we don't use MVDR, we can also use neural network to predict beamforming filters from spatial covariance matrices. This part is not implemented well yet, and not tested. 
    affine_layers = ReturnLayerIdxByName(layer, 'affine');
    tanh_layer = ReturnLayerIdxByName(layer, 'tanh');
    for i=affine_layers
        if i<tanh_layer && (para.topology.useWav && i>affine_layers(1))
            layer{i}.update = para.topology.updateBF;
        end
    end
end

if stage<=2; return; end        % we only build the network up to beamforming filter estimation. Not tested, so don't use this option. 

% below are code for integrating the beamforming front end with acoustic
% model back end. 

% set Mel filterbank linear transform
layer = InitMelLayer(layer, para);

% set global feature transform
splice_layer_idx = ReturnLayerIdxByName(layer, 'splice');
splice_layer_idx = splice_layer_idx(end);
if isfield(para.topology, 'initialAM_CE') && ~isempty(para.topology.initialAM_CE) % copy AM DNN parameters and preprocessing from an initial CE trained AM network
    pretrainDir = para.topology.initialAM_CE;
    tmp = load([pretrainDir '/final.feature_transform.mat']);
    layer{splice_layer_idx+1}.W = diag(tmp.kaldiNetwork{3}.transform);
    layer{splice_layer_idx+1}.b = layer{splice_layer_idx+1}.W*tmp.kaldiNetwork{2}.transform(:);
       
    modelfile = dir([pretrainDir '/*.nnet.*.mat']);
    tmp = load([pretrainDir '/' modelfile.name]);
    pretrainNetwork = tmp.kaldiNetwork;
    for i=1:length(para.topology.hiddenLayerSizeAM)+1
        layer{splice_layer_idx+i*2}.W = pretrainNetwork{i*2-1}.transform;
        layer{splice_layer_idx+i*2}.b = pretrainNetwork{i*2-1}.bias';
    end
elseif isfield(para.topology, 'initialAM_RBM') && ~isempty(para.topology.initialAM_RBM) % copy AM DNN parameters and preprocessing from an initial RBM trained AM network
    pretrainDir = para.topology.initialAM_RBM;
    tmp = load([pretrainDir '/final.feature_transform.mat']);
    layer{splice_layer_idx+1}.W = diag(tmp.kaldiNetwork{3}.transform);
    layer{splice_layer_idx+1}.b = layer{splice_layer_idx+1}.W*tmp.kaldiNetwork{2}.transform(:);
    
    tmp = load([pretrainDir '/6.dbn.txt.mat']);
    pretrainNetwork = tmp.kaldiNetwork;
    for i=1:length(para.topology.hiddenLayerSizeAM)
        layer{splice_layer_idx+i*2}.W = pretrainNetwork{i*2-1}.transform;
        layer{splice_layer_idx+i*2}.b = pretrainNetwork{i*2-1}.bias';
    end   
elseif isfield(para.topology, 'initialBF') && ~isempty(para.topology.initialBF)   % generate some fbank features and compute the global MVN required to normalize the fbanks. 
    layer1 = layer(1:splice_layer_idx);
    para1 = para;
    para1.out_layer_idx = length(layer1);
    [layer{splice_layer_idx+1}.W, layer{splice_layer_idx+1}.b] = computeGlobalCMVN(Data_tr, 100, para1, layer1);
else        % we have no information to initialize this layer
    layer(1:splice_layer_idx) = initializeParametersDNN_tree(layer(1:splice_layer_idx), ParseOptions2(para));
    layer1 = layer(1:splice_layer_idx);
    para1 = para;
    para1.out_layer_idx = length(layer1);
    [layer{splice_layer_idx+1}.W, layer{splice_layer_idx+1}.b] = computeGlobalCMVN(Data_tr, 100, para1, layer1);
    para1.out_layer_idx = length(layer1)+1;
    VerifyPreprocessingTree(layer(1:splice_layer_idx+1), Data_tr, para1, 100);
end

for i=1:length(para.topology.hiddenLayerSizeAM)+1
    layer{splice_layer_idx + i*2}.update = para.topology.updateAM;  % set the update field of acoustic model. 
end
end
