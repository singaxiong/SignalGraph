% This file serves as a template of defining the topology of the
% beamforming network with cross entropy training. 
% You should create a copy for each of your experiments and name them
% differently.
%
% References: 
% [1] Xiong Xiao, Shinji Watanabe, Hakan Erdogan, Liang Lu, John Hershey,
% Michael L. Seltzer, Guoguo Chen, Yu Zhang, Michael Mandel, Dong Yu, "Deep
% beamforming networks for multi-channel speech recognition", in ICASSP
% 2016.   
% [2] Xiong Xiao, Shinji Watanabe, Eng Siong Chng, Haizhou Li, "Beamforming
% Networks Using Spatial Covariance Features for Far-field Speech
% Recognition", in APSIPA 2016. 
% 
% Created by Xiong Xiao, Temasek Laboratories, NTU, Singapore.
% Last Modified: 29 Nov 2016
%
function para = ConfigMaskBFnetCE(para)
para.topology.fs = 16000;   % sampling rate
if isfield(para.topology, 'useChannel')
    if length(para.topology.useChannel)==1      % if useChannel is an integer, use it as the number of channels
        para.topology.nCh = para.topology.useChannel;
    else                                        % is useChannel is an integer array, use the specified channel indexes
        para.topology.nCh = length(para.topology.useChannel);
    end
else
    para.topology.nCh = 2;      % by default use 2 channels
end
para.topology = SetDefaultValue(para.topology, 'useFileName', 0);   % by default load waveforms into memory. If data is too big, we can also use wave file names
para.topology = SetDefaultValue(para.topology, 'nChMask', 1);   % by default use only 1 channel for mask prediction

% define the input type:
%   if set to 1, we will use raw waveform as input and need to define the
%   parameters for extracting features (such as GCC) and generating complex
%   Fourier transform of the input channels.
%   if set to 0, we will load precomputed features and Fourier
%   coefficients.
para.topology = SetDefaultValue(para.topology, 'useWav', 1);    % by default, we use waveform as input to the network

para.topology.fft_len = 512;
para.topology.freqBin = (0:1/para.topology.fft_len:0.5)*2*pi;
para.topology.nFreqBin = length(para.topology.freqBin);
if para.topology.useWav
    % define the parameters for extracting Fourier coefficients
    para.topology.frame_len = para.topology.fs * 0.025;
    para.topology.frame_shift = para.topology.fs * 0.01;
    para.topology.removeDC = 0;     % do not remove DC for faster speed. 
    para.topology.win_type = 'hamming';
end

% define the mask predicting subnet type
para.topology = SetDefaultValue(para.topology, 'MaskNetType', 'LSTM'); 
switch para.topology.MaskNetType
    case 'DNN'
        para.topology = SetDefaultValue(para.topology, 'hiddenLayerSizeMask', [1024]); 
        para.topology = SetDefaultValue(para.topology, 'contextSizeMask', 11);  % for DNN, we use 11 frames of consecutive frames
    case 'LSTM'
        para.topology = SetDefaultValue(para.topology, 'hiddenLayerSizeMask', [512]); 
        para.topology = SetDefaultValue(para.topology, 'useDeltaMask', 1);    % for LSTM, we use delta features without splicing
end

% define the BF subnet type
para.topology = SetDefaultValue(para.topology, 'BfNetType', 'MVDR');      % choose frame DNN or LSTM, default is DNN
switch para.topology.BfNetType
    case 'MVDR'     % use the MVDR formula to obtain beamforming weights
        
    case 'DNN'      % use DNN to predict beamforming weights
        para.topology = SetDefaultValue(para.topology, 'hiddenLayerSizeBF', [1024]); 
    case 'LSTM'
        para.topology = SetDefaultValue(para.topology, 'hiddenLayerSizeBF', [1024]); 
end
para.topology.bf_weight_dim = (para.topology.fft_len/2+1) * para.topology.nCh * 2;  % number of dimensions for weight matrix

% define the AM subnet type
para.topology = SetDefaultValue(para.topology, 'AmNetType', 'DNN');      % choose frame DNN or LSTM, default is DNN
switch para.topology.AmNetType
    case 'DNN'
        para.topology = SetDefaultValue(para.topology, 'hiddenLayerSizeAM', [1024]); 
    case 'LSTM'
        para.topology = SetDefaultValue(para.topology, 'hiddenLayerSizeAM', [1024]); 
end
para.topology = SetDefaultValue(para.topology, 'nSenone', 100);
para.topology = SetDefaultValue(para.topology, 'nFbank', 40);

% define the initialized subnets. Just put the path to the fields. Note
% that you can only set either initalAM_CE or initalAM_RBM.


% initial network for mask predicting subnet. 
para.topology = SetDefaultValue(para.topology, 'initialMask', '');
% initial network for beamforming predicting subnet. You need to make sure
% that the topology in the initialized network is the same as the topology
% of beamforming subnet defined above.
para.topology = SetDefaultValue(para.topology, 'initialBF', '');
% initial network for acoustic model trained by cross entropy
para.topology = SetDefaultValue(para.topology, 'initialAM_CE', '');
% initial network for acoustic model trained by RBM
para.topology = SetDefaultValue(para.topology, 'initialAM_RBM', '');

% tell the network whether we should update the Mask subnet, the BF subnet, the AM subnet,
% or all of them. 
para.topology = SetDefaultValue(para.topology, 'updateMask', 1);
para.topology = SetDefaultValue(para.topology, 'updateBF', 1);
para.topology = SetDefaultValue(para.topology, 'updateAM', 1);

end

