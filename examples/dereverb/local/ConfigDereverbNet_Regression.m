% This file serves as a template of defining the topology of the
% beamforming network with cross entropy training. 
% You should create a copy for each of your experiments and name them
% differently.
% 
% Created by Xiong Xiao, Temasek Laboratories, NTU, Singapore.
% Last Modified: 29 Nov 2016
%
function para = ConfigDereverbNet_Regression(para)
para.topology = SetDefaultValue(para.topology, 'fs', 16000); 
if isfield(para.topology, 'useChannel')
    para.topology.nCh = length(para.topology.useChannel);
else
    para.topology.nCh = 1;      % by default use 2 channels
end
para.topology = SetDefaultValue(para.topology, 'useFileName', 0);   % by default load waveforms into memory. If data is too big, we can also use wave file names

para.topology = SetDefaultValue(para.topology, 'fft_len', 512); 
para.topology.freqBin = (0:1/para.topology.fft_len:0.5)*2*pi;
para.topology.nFreqBin = length(para.topology.freqBin);
% define the parameters for extracting Fourier coefficients
para.topology.frame_len = para.topology.fs * 0.025;
para.topology.frame_shift = para.topology.fs * 0.01;
para.topology.removeDC = 0;     % do not remove DC for faster speed.
para.topology.win_type = 'hamming';

% define the regression network type
para.topology = SetDefaultValue(para.topology, 'RegressionNetType', 'LSTM'); 
switch para.topology.RegressionNetType
    case 'DNN'
        para.topology = SetDefaultValue(para.topology, 'hiddenLayerSize', [1024]); 
        para.topology = SetDefaultValue(para.topology, 'contextSize', 11);  % for DNN, we use 11 frames of consecutive frames
    case 'LSTM'
        para.topology = SetDefaultValue(para.topology, 'hiddenLayerSize', [512]); 
        para.topology = SetDefaultValue(para.topology, 'useDelta', 1);    % for LSTM, we use delta features without splicing
end

end

