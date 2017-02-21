% This file create a simple regression based network for speech
% dereverberation or enhancement
%
% Created by Xiong Xiao, Temasek Laboratories, NTU, Singapore.
% Last Modified: 08 Feb 2017
%
function layer = genNetworkDereverb_RegressionComplex(para)
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

outputDim = nFreqBin*2;

switch para.RegressionNetType
    case 'DNN'
        layer{end+1}.name = 'Splice';
        layer{end}.prev = -1;
        layer{end}.context = para.contextSize;
        layer{end}.dim = [layer{end}.context 1]*layer{length(layer)+layer{end}.prev}.dim(1);
        layer{end}.skipBP = 1;

        layerRegression = genNetworkFeedForward_v2(layer{end}.dim(1), para.hiddenLayerSize, outputDim, 'mse', 'linear');
    case 'LSTM'
%         layer{end+1}.name = 'Delta';
%         layer{end}.prev = -1;
%         layer{end}.delta_order = 2;
%         layer{end}.dim = [layer{end}.delta_order+1 1]*layer{length(layer)+layer{end}.prev}.dim(1);
%         layer{end}.skipBP = 1;
%         
%         layer{end+1}.name = 'Affine';       % scaling the Fourier transform
%         layer{end}.prev = -1;
%         layer{end}.W = [];
%         layer{end}.b = [];
%         layer{end}.dim = [1 1] * layer{length(layer)+layer{end}.prev}.dim(1);
%         layer{end}.update = 0;

        tmpTopology.inputDim = layer{end}.dim(1);
        tmpTopology.hiddenLayerSizeLSTM = para.hiddenLayerSize;
        tmpTopology.usePastState = zeros(1,length(para.hiddenLayerSize)); % do not use peeping hole
        tmpTopology.hiddenLayerSizeFF = [];
        tmpTopology.outputDim = outputDim;
        tmpTopology.costFn = 'mse';
        tmpTopology.LastActivation4MSE = 'linear';
        layerRegression = genNetworkLSTM(tmpTopology);
end
layer = [layer layerRegression(2:end-2)];
MSE_layer = layerRegression(end);

if ~isempty(para.hiddenLayerSizeFF)
    layerFF = genNetworkFeedForward_v2(para.hiddenLayerSize(end), para.hiddenLayerSizeFF, outputDim, 'mse', 'linear');
    layer = [layer(1:end-1) layerFF(2:end-2)];
end

% layer{end+1}.name = 'realImag2complex';
% layer{end}.prev = -1;
% layer{end}.dim = [0.5 1] * layer{end-1}.dim(1);

output_idx = length(layer);

idx = ReturnLayerIdxByName(layer, 'mu_law');
CleanLayer = layer(1:idx(1));
CleanLayer{1}.inputIdx = 2;
CleanLayer{1}.dim(:) = 1;
CleanLayer{2}.dim(:) = CleanLayer{2}.dim(:)/para.nCh;

layer = [layer CleanLayer MSE_layer];
layer{end}.prev = [output_idx-length(layer) -1];
layer{end}.dim(2) = layer{end-1}.dim(1);

layer = FinishLayer(layer);
end
