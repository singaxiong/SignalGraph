% This file create a simple regression based network for speech
% dereverberation or enhancement
%
% Created by Xiong Xiao, Temasek Laboratories, NTU, Singapore.
% Last Modified: 08 Feb 2017
%
function layer = genNetworkDereverb_Gaussian(para, stage)
para.freqBin = (0:1/para.fft_len:0.5)*2*pi; % w = 2*pi*f, where f is the normalized frequency k/N is from 0 to 0.5.
nFreqBin = length(para.freqBin);

% Part 1: generate the BF weight predicting subnet

if para.useWav  % input is row waveform
    layer{1}.name = 'Input';
    layer{end}.inputIdx = 1;
    layer{end}.dim = [1 1]*para.nCh;
    
    layer{end+1}.name = 'stft';
    layer{end}.prev = -length(layer)+1;
    layer{end}.fft_len = para.fft_len;
    layer{end}.frame_len = para.frame_len;
    layer{end}.frame_shift = para.frame_shift;
    layer{end}.removeDC = para.removeDC;
    layer{end}.win_type = para.win_type;
    layer{end}.dim = [(para.fft_len/2+1)*para.nCh layer{length(layer)+layer{end}.prev}.dim(1)];
    layer{end}.skipBP = 1;  % skip backpropagation
    
    layer{end+1}.name = 'Affine';       % scaling the Fourier transform
    layer{end}.prev = -1;
    layer{end}.W = [];
    layer{end}.b = [];
    layer{end}.dim = [1 1] * layer{length(layer)+layer{end}.prev}.dim(1);
    layer{end}.update = 0;
    layer{end}.skipBP = 1;  % skip backpropagation
    
    % extract a subset of dimensions for prediction. Not used currently, so we
    % will use all the dimensions.
    layer{end+1}.name = 'ExtractDims';
    layer{end}.prev = -1;
    layer{end}.dimIndex = 1:nFreqBin*para.nCh;
    layer{end}.dim = [length(layer{end}.dimIndex) layer{length(layer)+layer{end}.prev}.dim(1)];
    layer{end}.skipBP = 1;
    
    % get the log power spectrum and perform CMN
    layer{end+1}.name = 'Power';
    layer{end}.prev = -1;
    layer{end}.dim = [1 1] * layer{end-1}.dim(1);
    layer{end}.skipBP = 1;
    
    layer{end+1}.name = 'Log';
    layer{end}.const = 1e-2;
    layer{end}.prev = -1;
    layer{end}.dim = [1 1]*layer{end-1}.dim(1);
    layer{end}.skipBP = 1;
else    % input is log spectrogram
    layer{1}.name = 'Input';
    layer{end}.inputIdx = 1;
    layer{end}.dim = [1 1]*(para.fft_len/2+1)*para.nCh;
end

if para.useCMN
    layer{end+1}.name = 'CMN';
    layer{end}.prev = -1;
    layer{end}.dim = [1 1]*layer{end-1}.dim(1);
    layer{end}.skipBP = 1;
end

layer{end+1}.name = 'Delta';
layer{end}.prev = -1;
layer{end}.delta_order = 2;
layer{end}.dim = [layer{end}.delta_order+1 1]*layer{length(layer)+layer{end}.prev}.dim(1);
layer{end}.skipBP = 1;

layer{end+1}.name = 'Affine';       % scaling the Fourier transform
layer{end}.prev = -1;
layer{end}.W = [];
layer{end}.b = [];
layer{end}.dim = [1 1] * layer{length(layer)+layer{end}.prev}.dim(1);
layer{end}.update = 0;
layer{end}.skipBP= 1;

affineLayer = length(layer);

if ~isempty(para.hiddenLayerSizeShared)
    tmpTopology.inputDim = layer{end}.dim(1);
    tmpTopology.hiddenLayerSizeLSTM = para.hiddenLayerSizeShared;
    tmpTopology.usePastState = zeros(1,length(para.hiddenLayerSizeShared)); % do not use peeping hole
    tmpTopology.hiddenLayerSizeFF = [];
    tmpTopology.outputDim = 1024;
    tmpTopology.useAffineBtwLSTM = 1;
    tmpTopology.costFn = 'mse';
    tmpTopology.LastActivation4MSE = 'linear';
    layerShared = genNetworkLSTM(tmpTopology);
    if stage==2.2   % for stage 2.2, we don't even update the shared layer, only update the var subnet
        for i=1:length(layerShared)
            layerShared{i}.update = 0;
            layerShared{i}.skipBP = 1;
        end
    end
    layer = [layer layerShared(2:end-3)];
end

shared_lstm_idx = length(layer);

% get the mean 
tmpTopology.inputDim = layer{shared_lstm_idx}.dim(1);
tmpTopology.hiddenLayerSizeLSTM = para.hiddenLayerSizeMu;
tmpTopology.usePastState = zeros(1,length(para.hiddenLayerSizeMu)); % do not use peeping hole
tmpTopology.hiddenLayerSizeFF = [];
if strcmpi(para.DeltaGenerationType, 'DeltaByEqn')
    tmpTopology.outputDim = nFreqBin;
else
    tmpTopology.outputDim = nFreqBin*3;
end
tmpTopology.costFn = 'mse';
tmpTopology.LastActivation4MSE = 'linear';
layerMu = genNetworkLSTM(tmpTopology);
layerMu = layerMu(2:end-2);
if ~isempty(para.hiddenLayerSizeMu) && ~isempty(para.hiddenLayerSizeShared)  % add a projection layer between two LSTM layers
    layerMu = [layer(affineLayer) layerMu];
    layerMu{1}.update = 1;
    layerMu{1}.skipBP = 0;
    layerMu{1}.dim = [1 1]*layer{shared_lstm_idx}.dim(1);
end
layerMu{1}.prev = shared_lstm_idx - length(layer)-1;
if floor(stage)==2
    for i=1:length(layerMu)
        layerMu{i}.update = 0;
        if stage==2.2  % if there is no shared layers, we don't let the gradient from mean layers
            layerMu{i}.skipBP = 1;
        end
    end
end
layer = [layer layerMu];

if strcmpi(para.DeltaGenerationType, 'DeltaByEqn')
    layer{end+1}.name = 'Delta';
    layer{end}.prev = -1;
    layer{end}.delta_order = 2;
    layer{end}.dim = [layer{end}.delta_order+1 1]*layer{length(layer)+layer{end}.prev}.dim(1);
end

mean_idx = length(layer);

% get the var
tmpTopology.inputDim = layer{shared_lstm_idx}.dim(1);
tmpTopology.hiddenLayerSizeLSTM = para.hiddenLayerSizeVar;
tmpTopology.usePastState = zeros(1,length(para.hiddenLayerSizeVar)); % do not use peeping hole
tmpTopology.hiddenLayerSizeFF = [];
tmpTopology.outputDim = nFreqBin*3;
tmpTopology.costFn = 'mse';
tmpTopology.LastActivation4MSE = 'linear';
layerVar = genNetworkLSTM(tmpTopology);
layerVar = layerVar(2:end-2);
layerVar{1}.prev = shared_lstm_idx - length(layer)-1;
if stage==1
    for i=1:length(layerVar)
        layerVar{i}.update = 0;
        layerVar{i}.skipBP = 1;
    end
    % always output unit variance
    layerVar{end}.W = zeros(layerVar{end}.dim(1), layerVar{end}.dim(2));
    layerVar{end}.b = zeros(layerVar{end}.dim(1),1);
    layerVar{end}.b(nFreqBin+1:nFreqBin*2) = -log(para.MSECostWeightSDA(2)^2);
    layerVar{end}.b(nFreqBin*2+1:nFreqBin*3) = -log(para.MSECostWeightSDA(3)^2);
end
layer = [layer layerVar];

layer{end+1}.name = 'exp';
layer{end}.prev = -1;
layer{end}.dim = [1 1]* layer{end-1}.dim(1);

var_idx = length(layer);

if para.useWav
    log_idx = ReturnLayerIdxByName(layer, 'log');
    if para.useCMN
        CleanLayer = layer(1:log_idx(1)+1);
    else
        CleanLayer = layer(1:log_idx(1));
    end
    CleanLayer{1}.inputIdx = 2;
    CleanLayer{1}.dim(:) = 1;
    CleanLayer{2}.dim(:) = CleanLayer{2}.dim(:)/para.nCh;
    CleanLayer{3}.dim(:) = CleanLayer{3}.dim(:)/para.nCh;
    CleanLayer(4) = [];
else
    CleanLayer{1}.name = 'Input';
    CleanLayer{end}.inputIdx = 2;
    CleanLayer{end}.dim = [1 1]*(para.fft_len/2+1);
end

CleanLayer{end+1}.name = 'Delta';
CleanLayer{end}.prev = -1;
CleanLayer{end}.delta_order = 2;
CleanLayer{end}.dim = [CleanLayer{end}.delta_order+1 1]*CleanLayer{length(CleanLayer)+CleanLayer{end}.prev}.dim(1);

layer = [layer CleanLayer];

layer{end+1}.name = 'LL_Gaussian';
layer{end}.prev = [mean_idx-length(layer) var_idx-length(layer) -1];
layer{end}.dim = [1 layer{end-1}.dim(1)];

layer = FinishLayer(layer);
end
