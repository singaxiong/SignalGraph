% This file create the joint mask predicting, beamforming weight
% predicting, and acoustic model networks consists of a linked list
% of computing layers. 
%
% References: 
% 
% Created by Xiong Xiao, Temasek Laboratories, NTU, Singapore.
% Last Modified: 08 Jul 2016
%
function layer = genNetworkMaskBF_CE(para)
para.freqBin = (0:1/para.fft_len:0.5)*2*pi; % w = 2*pi*f, where f is the normalized frequency k/N is from 0 to 0.5.
nFreqBin = length(para.freqBin);
BFWeightDim = length(para.freqBin)*para.nCh*2;

% Part 1: generate the BF weight predicting subnet

layer{1}.name = 'Input';
layer{end}.inputIdx = 1;
layer{end}.dim = [1 1]*para.nCh;
if para.useWav  % we should use waveform as input
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
end

% extract the first channel to predict the mask. We can later modify the
% network to predict the mask for all channels and then use pooling. 
layer{end+1}.name = 'ExtractDims';
layer{end}.prev = -1;
layer{end}.dimIndex = 1:nFreqBin*para.nChMask;
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

switch para.MaskNetType
    case 'DNN'
        layer{end+1}.name = 'Splice';
        layer{end}.prev = -1;
        layer{end}.context = para.contextSizeMask;
        layer{end}.dim = [layer{end}.context 1]*layer{length(layer)+layer{end}.prev}.dim(1);
        layer{end}.skipBP = 1;

        layerMask = genNetworkFeedForward_v2(layer{end}.dim(1), para.hiddenLayerSizeMask, nFreqBin, 'mse', 'sigmoid');
    case 'LSTM'
        layer{end+1}.name = 'Delta';
        layer{end}.prev = -1;
        layer{end}.delta_order = 2;
        layer{end}.dim = [layer{end}.delta_order+1 1]*layer{length(layer)+layer{end}.prev}.dim(1);
        layer{end}.skipBP = 1;

        tmpTopology.inputDim = layer{end}.dim(1);
        tmpTopology.hiddenLayerSizeLSTM = para.hiddenLayerSizeMask;
        tmpTopology.usePastState = zeros(1,length(para.hiddenLayerSizeMask)); % do not use peeping hole
        tmpTopology.hiddenLayerSizeFF = [];
        tmpTopology.outputDim = nFreqBin;
        tmpTopology.costFn = 'mse';
        tmpTopology.LastActivation4MSE = 'sigmoid';
        layerMask = genNetworkLSTM(tmpTopology);
        layerMask = layerMask(2:end-2);
end
% add the global MVN to the input to mask predicting subnet
layer{end+1}.name = 'Affine';   % diagonal transofrm
layer{end}.prev = -1;
layer{end}.W = []; % to be initialized randomly or by pretraining
layer{end}.b = [];
layer{end}.dim = [1 1]*layer{length(layer)+layer{end}.prev}.dim(1);
layer{end}.update = 0;
layer{end}.skipBP = 1;

% concatenate layer with layerMask
layer = [layer layerMask];
layer{end}.next = 1;

% define the beamforming weight prediciting subnet
% first, get the spatial covariance of noise and speech

layer{end+1}.name = 'SpatialCovMask';
stft_idx = ReturnLayerIdxByName(layer, 'stft');
layer{end}.prev = [-1 stft_idx(1)+1-length(layer)];
layer{end}.dim = [2*para.nCh^2 1]*nFreqBin;

switch para.BfNetType
    case 'MVDR'
        layer{end+1}.name = 'MVDR_SpatialCov';
        layer{end}.prev = -1;
        layer{end}.fs = para.fs;
        layer{end}.freqBin = para.freqBin;
        layer{end}.dim = [para.nCh 2*para.nCh^2]*nFreqBin;
    case 'DNN'  % we can also predict filter weights from covariance matrix. Never tried. May have bugs. 
        layerBF = genNetworkFeedForward_v2(layer{end}.dim(1), para.hiddenLayerSizeBF, BFWeightDim, 'mse', 'tanh');
    case 'LSTM'
        % to be implemented
end
if strcmpi(para.BfNetType, 'DNN') || strcmpi(para.BfNetType, 'LSTM')
    layer = [layer layerBF(2:end-2)];   % remove the last 2 layers
    layer{end}.next = 1;
    
    layer{end+1}.name = 'real_imag2BFweight';
    layer{end}.prev = -1;
    layer{end}.fs = para.fs;
    layer{end}.freqBin = para.freqBin;
    layer{end}.dim = [1 2] * BFWeightDim/2;
end

% Part 2: generate the BF and feature extraction subnet

layer{end+1}.name = 'Beamforming';
stft_idx = ReturnLayerIdxByName(layer, 'stft');
layer{end}.prev = [-1 stft_idx+1-length(layer)];
layer{end}.freqBin = para.freqBin;
layer{end}.dim = [1 para.nCh] * nFreqBin;

layer{end+1}.name = 'Power';
layer{end}.prev = -1;
layer{end}.dim = [1 1] * nFreqBin;

layer{end+1}.name = 'Mel';
layer{end}.prev = -1;
layer{end}.W = [];
layer{end}.b = [];
layer{end}.dim = [para.nFbank nFreqBin];
layer{end}.update = 0;

layer{end+1}.name = 'Log';
layer{end}.const = 1e-2;
layer{end}.prev = -1;
layer{end}.dim = [1 1]*layer{end-1}.dim(1);

layer{end+1}.name = 'CMN';
layer{end}.prev = -1;
layer{end}.dim = [1 1]*layer{end-1}.dim(1);

layer{end+1}.name = 'Delta';
layer{end}.prev = -1;
layer{end}.delta_order = 2;
layer{end}.dim = [layer{end}.delta_order+1 1]*layer{length(layer)+layer{end}.prev}.dim(1);
layer{end}.update = 0;

layer{end+1}.name = 'Splice';
layer{end}.prev = -1;
layer{end}.context = 11;
layer{end}.dim = [layer{end}.context 1]*layer{length(layer)+layer{end}.prev}.dim(1);
layer{end}.update = 0;

layer{end+1}.name = 'Affine';   % diagonal transofrm
layer{end}.prev = -1;
layer{end}.W = []; % to be initialized randomly or by pretraining
layer{end}.b = [];
layer{end}.dim = [1 1]*layer{length(layer)+layer{end}.prev}.dim(1);
layer{end}.update = 0;

% Part 3: generate the AM subnet

switch para.AmNetType
    case 'DNN'
        layerAM = genNetworkFeedForward_v2(layer{end}.dim(1), para.hiddenLayerSizeAM, para.nSenone, 'cross_entropy');
    case 'LSTM'
        % to be implemented
end
layerAM{end-1}.inputIdx = 2;    % if use raw waveform as input, the label will be at the second stream

layer = [layer layerAM(2:end)];

layer = FinishLayer(layer);
end
