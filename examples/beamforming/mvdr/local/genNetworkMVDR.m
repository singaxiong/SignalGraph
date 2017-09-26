% This function create a network of MVDR beamforming. You need to supply
% the noise and speech masks. 
% 
% Created by Xiong Xiao
% Last Modified: 30 Jun 2017
%
function layer = genNetworkMVDR(para)
para.freqBin = (0:1/para.fft_len:0.5)*2*pi; % w = 2*pi*f, where f is the normalized frequency k/N is from 0 to 0.5.
nFreqBin = length(para.freqBin);
BFWeightDim = length(para.freqBin)*para.nCh*2;

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

layer{end+1}.name = 'weight2activation';
layer{end}.dim = [];

layer{end+1}.name = 'weight2activation';
layer{end}.dim = [];

layer{end+1}.name = 'SpatialCovSplitMask';
stft_idx = ReturnLayerIdxByName(layer, 'stft');
layer{end}.prev = [-2 -1 stft_idx(1)-length(layer)];
layer{end}.dim = [2*para.nCh^2 1]*nFreqBin;

layer{end+1}.name = 'MVDR_SpatialCov';
layer{end}.prev = -1;
layer{end}.fs = para.fs;
layer{end}.freqBin = para.freqBin;
layer{end}.dim = [para.nCh 2*para.nCh^2]*nFreqBin;

layer{end+1}.name = 'Beamforming';
stft_idx = ReturnLayerIdxByName(layer, 'stft');
layer{end}.prev = [-1 stft_idx-length(layer)];
layer{end}.freqBin = para.freqBin;
layer{end}.dim = [1 para.nCh] * nFreqBin;

layer = FinishLayer(layer);
end
