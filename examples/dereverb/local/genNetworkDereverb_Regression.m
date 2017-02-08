% This file create a simple regression based network for speech
% dereverberation or enhancement
%
% Created by Xiong Xiao, Temasek Laboratories, NTU, Singapore.
% Last Modified: 08 Feb 2017
%
function layer = genNetworkDereverb_Regression(para)
para.freqBin = (0:1/para.fft_len:0.5)*2*pi; % w = 2*pi*f, where f is the normalized frequency k/N is from 0 to 0.5.
nFreqBin = length(para.freqBin);

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

layer{end+1}.name = 'CMN';
layer{end}.prev = -1;
layer{end}.dim = [1 1]*layer{end-1}.dim(1);
layer{end}.skipBP = 1;

switch para.RegressionNetType
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

        layerRegression = genNetworkFeedForward_v2(layer{end}.dim(1), para.hiddenLayerSize, nFreqBin*3, 'mse', 'linear');
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
        tmpTopology.outputDim = nFreqBin*3;
        tmpTopology.costFn = 'mse';
        tmpTopology.LastActivation4MSE = 'linear';
        layerRegression = genNetworkLSTM(tmpTopology);
end
layer = [layer layerRegression(2:end)];

output_idx = length(layer)-2;

cmn_idx = ReturnLayerIdxByName(layer, 'cmn');
reusable_layer = layer(1:cmn_idx(1));
reusable_layer{1}.inputIdx = 2;
reusable_layer{1}.dim(:) = 1;
reusable_layer{2}.dim(:) = reusable_layer{2}.dim(:)/para.nCh;
reusable_layer{3}.dim(:) = reusable_layer{3}.dim(:)/para.nCh;
reusable_layer(4) = [];
reusable_layer{end+1}.name = 'Delta';
reusable_layer{end}.prev = -1;
reusable_layer{end}.delta_order = 2;
reusable_layer{end}.dim = [reusable_layer{end}.delta_order+1 1]*reusable_layer{length(reusable_layer)+reusable_layer{end}.prev}.dim(1);

reusable_layer{end+1}.name = 'Affine';       % scaling the contribution of output dimensions to MSE cost function.
reusable_layer{end}.prev = -1;
reusable_layer{end}.W = [];
reusable_layer{end}.b = [];
reusable_layer{end}.dim = [1 1] * reusable_layer{length(reusable_layer)+reusable_layer{end}.prev}.dim(1);
reusable_layer{end}.update = 0;

layer = [layer(1:end-2) reusable_layer layer(end)];
layer{end}.prev = [output_idx-length(layer) -1];

layer = FinishLayer(layer);
end
