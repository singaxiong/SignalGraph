% This file create a simple regression based network for speech
% dereverberation or enhancement
%
% Created by Xiong Xiao, Temasek Laboratories, NTU, Singapore.
% Last Modified: 08 Feb 2017
%
function layer = genNetworkDereverb_FilterPrediction(para, type)
para.freqBin = (0:1/para.fft_len:0.5)*2*pi; % w = 2*pi*f, where f is the normalized frequency k/N is from 0 to 0.5.
nFreqBin = length(para.freqBin);
nWeights = para.FilterTempContext * para.FilterFreqContext * nFreqBin * 2;

% Part 1: generate the BF weight predicting subnet

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

stft_idx = length(layer);

% extract a subset of dimensions for prediction. Not used currently, so we
% will use all the dimensions.
layer{end+1}.name = 'ExtractDims';
layer{end}.prev = -1;
layer{end}.dimIndex = 1:nFreqBin*para.nCh;
layer{end}.dim = [length(layer{end}.dimIndex) layer{length(layer)+layer{end}.prev}.dim(1)];
layer{end}.skipBP = 1;

if para.useComplexFeature
    layer{end+1}.name = 'complex2realImag';
    layer{end}.prev = -1;
    layer{end}.dim = [2 1]* layer{end-1}.dim(1);
    layer{end}.skipBP = 1;
    
    layer{end+1}.name = 'absmax_norm';
    layer{end}.prev = -1;
    layer{end}.dim = [1 1] * layer{end-1}.dim(1);
    layer{end}.skipBP = 1;
    
    layer{end+1}.name = 'mu_law';
    layer{end}.prev = -1;
    layer{end}.mu = para.mu_law_factor;
    layer{end}.dim = [1 1] * layer{end-1}.dim(1);
    layer{end}.skipBP = 1;
    
    layer{end+1}.name = 'Affine';       % scaling the Fourier transform
    layer{end}.prev = -1;
    layer{end}.W = [];
    layer{end}.b = [];
    layer{end}.dim = [1 1] * layer{length(layer)+layer{end}.prev}.dim(1);
    layer{end}.update = 0;
    layer{end}.skipBP = 1;
else
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
    
    if para.useCMN
        layer{end+1}.name = 'CMN';
        layer{end}.prev = -1;
        layer{end}.dim = [1 1]*layer{end-1}.dim(1);
        layer{end}.skipBP = 1;
    end
end

switch para.FilterNetType
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

        layerRegression = genNetworkFeedForward_v2(layer{end}.dim(1), para.hiddenLayerSize, nWeights, 'mse', 'linear');
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
        tmpTopology.outputDim = nWeights;
        tmpTopology.costFn = 'mse';
        tmpTopology.LastActivation4MSE = 'linear';
        layerRegression = genNetworkLSTM(tmpTopology);
end
layer = [layer layerRegression(2:end-2)];
MSE_layer = layerRegression(end);

if ~isempty(para.hiddenLayerSizeFF)
    layerFF = genNetworkFeedForward_v2(para.hiddenLayerSize(end), para.hiddenLayerSizeFF, nWeights, 'mse', 'linear');
    layer = [layer layerFF(2:end-2)];
end

if strcmpi(type, 'subnet')
    layer{end+1}.name = 'Input';
    layer{end}.inputIdx = 2;
    layer{end}.dim = [1 1]*layer{end-1}.dim(1);
  
    layer = [layer MSE_layer];

    layer = FinishLayer(layer);
    return;
end

if 1
    layer{end+1}.name = 'mean';
    layer{end}.prev = -1;
    layer{end}.pool_idx = 2;
    layer{end}.dim = [1 1] * layer{end-1}.dim(1);
else
    layer{end+1}.name = 'frame_select';
    layer{end}.prev = -1;
    layer{end}.frameSelect = 'last';
    layer{end}.dim = [1 1] * layer{end-1}.dim(1);    
end

% layer{end+1}.name = 'tanh';
% layer{end}.prev = -1;
% layer{end}.dim = [1 1] * layer{end-1}.dim(1);

layer{end+1}.name = 'realImag2complex';
layer{end}.prev = -1;
layer{end}.dim = [0.5 1] * layer{end-1}.dim(1);

layer{end+1}.name = 'copyVec2Mat';
layer{end}.prev = -1;
mask = eye(para.nFreqBin);
mask = repmat(mask, 1, para.FilterTempContext);
layer{end}.index2copy = find(mask==1);     % define the linear index in the matrix, that the vector's elements will be copied to. 
layer{end}.targetDims = [1 para.FilterTempContext] * para.nFreqBin;
layer{end}.dim = [para.nFreqBin layer{end-1}.dim(1)];

layer{end+1}.name = 'permute';
layer{end}.prev = -1;
layer{end}.permute_order = [1 2 4 3];
layer{end}.dim = [1 1]* layer{end-1}.dim(1);

layer{end+1}.name = 'Splice';
layer{end}.prev = stft_idx - length(layer);
layer{end}.context = para.FilterTempContext;
layer{end}.dim = [layer{end}.context 1]*layer{length(layer)+layer{end}.prev}.dim(1);
layer{end}.skipBP = 1;

layer{end+1}.name = 'matrix_multiply';
layer{end}.prev = [-2 -1];
layer{end}.dim = [1 para.FilterTempContext] * para.nFreqBin;
layer{end}.update = 1;

stft2logspec{1}.name = 'Power';
stft2logspec{end}.prev = -1;
stft2logspec{end}.dim = [1 1] * layer{end}.dim(1);

stft2logspec{end+1}.name = 'Log';
stft2logspec{end}.const = 1e-2;
stft2logspec{end}.prev = -1;
stft2logspec{end}.dim = [1 1]*stft2logspec{end-1}.dim(1);

if para.useCMN
    stft2logspec{end+1}.name = 'CMN';
    stft2logspec{end}.prev = -1;
    stft2logspec{end}.dim = [1 1]*stft2logspec{end-1}.dim(1);
end

stft2logspec{end+1}.name = 'Delta';
stft2logspec{end}.prev = -1;
stft2logspec{end}.delta_order = 2;
stft2logspec{end}.dim = [stft2logspec{end}.delta_order+1 1]*stft2logspec{length(stft2logspec)+stft2logspec{end}.prev}.dim(1);

stft2logspec{end+1}.name = 'Affine';       % scaling the contribution of output dimensions to MSE cost function.
stft2logspec{end}.prev = -1;
stft2logspec{end}.W = [];
stft2logspec{end}.b = [];
stft2logspec{end}.dim = [1 1] * stft2logspec{length(stft2logspec)+stft2logspec{end}.prev}.dim(1);
stft2logspec{end}.update = 0;

layer = [layer stft2logspec];

output_idx = length(layer);

idx = ReturnLayerIdxByName(layer, 'ExtractDims');
reusable_layer = layer(1:idx(1)-1);

reusable_layer{1}.inputIdx = 2;
reusable_layer{1}.dim(:) = 1;
reusable_layer{2}.dim(:) = reusable_layer{2}.dim(:)/para.nCh;
reusable_layer{3}.dim(:) = reusable_layer{3}.dim(:)/para.nCh;

reusable_layer = [reusable_layer stft2logspec];

for i=1:length(reusable_layer)
    reusable_layer{i}.skipBP = 1;
end

layer = [layer reusable_layer MSE_layer];
layer{end}.prev = [output_idx-length(layer) -1];

layer = FinishLayer(layer);
end
