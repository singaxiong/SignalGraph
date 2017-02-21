% This file create a simple regression based network for speech
% dereverberation or enhancement
%
% Created by Xiong Xiao, Temasek Laboratories, NTU, Singapore.
% Last Modified: 08 Feb 2017
%
function layer = genNetworkDereverb_Masking(para)
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

logSpec_idx = length(layer);
outputDim = nFreqBin;

switch para.MaskNetType
    case 'DNN'
        layer{end+1}.name = 'Splice';
        layer{end}.prev = -1;
        layer{end}.context = para.contextSize;
        layer{end}.dim = [layer{end}.context 1]*layer{length(layer)+layer{end}.prev}.dim(1);
        layer{end}.skipBP = 1;
        
        layer{end+1}.name = 'Affine';       % scaling the Fourier transform
        layer{end}.prev = -1;
        layer{end}.W = [];
        layer{end}.b = [];
        layer{end}.dim = [1 1] * layer{length(layer)+layer{end}.prev}.dim(1);
        layer{end}.update = 0;

        layerRegression = genNetworkFeedForward_v2(layer{end}.dim(1), para.hiddenLayerSize, outputDim, 'mse', 'sigmoid');
    case 'LSTM'
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

        tmpTopology.inputDim = layer{end}.dim(1);
        tmpTopology.hiddenLayerSizeLSTM = para.hiddenLayerSize;
        tmpTopology.usePastState = zeros(1,length(para.hiddenLayerSize)); % do not use peeping hole
        tmpTopology.hiddenLayerSizeFF = [];
        tmpTopology.outputDim = outputDim;
        tmpTopology.costFn = 'mse';
        tmpTopology.LastActivation4MSE = 'sigmoid';
        layerRegression = genNetworkLSTM(tmpTopology);
end
layer = [layer layerRegression(2:end-2)];
MSE_layer = layerRegression{end};

if ~isempty(para.hiddenLayerSizeFF)
    layerFF = genNetworkFeedForward_v2(para.hiddenLayerSize(end), para.hiddenLayerSizeFF, outputDim, 'mse', 'linear');
    layer = [layer(1:end-1) layerFF(2:end-2)];
end

layer{end}.dim(1) = nFreqBin;

% apply the masking
if para.useWav  % input is row waveform
    power_idx = ReturnLayerIdxByName(layer, 'power');
    layer{end+1}.name = 'hadamard';
    layer{end}.dim = [1 1]*layer{end-1}.dim(1);
    layer{end}.prev = [-1 power_idx(1)-length(layer)];
else
    input_idx = ReturnLayerIdxByName(layer, 'input');
    layer{end+1}.name = 'exp';
    layer{end}.prev = input_idx(1)-length(layer);
    layer{end}.dim = [1 1]*layer{input_idx(1)}.dim(1);

    layer{end+1}.name = 'hadamard';
    layer{end}.dim = [1 1]*layer{end-1}.dim(1);
    layer{end}.prev = [-1 -2];
end

layer{end+1}.name = 'Log';
layer{end}.const = 1e-2;
layer{end}.prev = -1;
layer{end}.dim = [1 1]*layer{end-1}.dim(1);

if para.useCMN
    layer{end+1}.name = 'CMN';
    layer{end}.prev = -1;
    layer{end}.dim = [1 1]*layer{end-1}.dim(1);
    layer{end}.skipBP = 1;
end

DeltaLayer{1}.name = 'Delta';
DeltaLayer{end}.prev = -1;
DeltaLayer{end}.delta_order = 2;
DeltaLayer{end}.dim = [DeltaLayer{end}.delta_order+1 1]*nFreqBin;

DeltaLayer{end+1}.name = 'Affine';       % scaling the contribution of output dimensions to MSE cost function.
DeltaLayer{end}.prev = -1;
DeltaLayer{end}.W = [];
DeltaLayer{end}.b = [];
DeltaLayer{end}.dim = [1 1] * DeltaLayer{length(DeltaLayer)+DeltaLayer{end}.prev}.dim(1);
DeltaLayer{end}.update = 0;

layer = [layer DeltaLayer];


output_idx = length(layer);

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

CleanLayer{end+1}.name = 'Affine';       % scaling the contribution of output dimensions to MSE cost function.
CleanLayer{end}.prev = -1;
CleanLayer{end}.W = [];
CleanLayer{end}.b = [];
CleanLayer{end}.dim = [1 1] * CleanLayer{length(CleanLayer)+CleanLayer{end}.prev}.dim(1);
CleanLayer{end}.update = 0;
for i=1:length(CleanLayer)
    CleanLayer{i}.skipBP = 1;
end

layer = [layer CleanLayer MSE_layer];
layer{end}.prev = [output_idx-length(layer) -1];

layer = FinishLayer(layer);
end
