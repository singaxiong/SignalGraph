% This script train an LSTM based speech mask predictor, using simulated
% data of CHiME-4 challenge.
% Xiong Xiao, Nanyang Technological University, Singapore
% Last modified: 30 Nov 2016. 
clear
addpath('local', '../lib');
% define SignalGraph settings. Type ParseOptions2() in command line to see
% the options. 
para.IO.nStream = 2;        % two data streams, one is input signal, one is mask label
para.NET.sequential = 1;    % do not randomize at frame level, randomize at sentence level
para.NET.variableLengthMinibatch = 1;   % a minibatch may contain multiple sentences of different lengths
para.NET.nSequencePerMinibatch = 20;    % number of sentences per minibatch
para.NET.maxNumSentInBlock = 100;       % maximum number of sentences in a block
para.NET.L2weight = 3e-4;
para.NET.learning_rate = 3e-2;
para.NET.learning_rate_decay_rate = 0.999;
para.NET.momentum = [0.9];
para.NET.gradientClipThreshold = 1;
para.NET.weight_clip = 10;
para.useGPU     = 0;                  % don't use GPU for single sentence minibatch for LSTM. It's even slower than CPU. 
para.minItr     = 10;
para.displayGPUstatus = 1;
para.displayInterval = 1;
para.skipInitialEval = 1;
para.displayTag = 'LSTM-Mask';

% define network topology
para.topology.useFileName = 0;      % if set to 0, load all training data to memory. otherwise, only load file names. 
para.topology.useWav = 1;           % if set to 1, load waveforms and compute log spectrogram features in the network (save space); otherwise, load log spectrogram features directly. 
para.topology.useChannel = 1;       % choose which channel [1-6] to train the LSTM. 
para.topology.nChMask = 1;          % 
para.topology.MaskNetType = 'LSTM'; % define mask subnet type
para.topology.BfNetType = 'MVDR';   % define beamforming subnet type
para.topology.AmNetType = 'DNN';    % define acoustic model subnet type
para.topology.initialMask = '';     % provide a file name for initial mask if any. 
para.topology.nSenone = 3968;       % number of senones in the acoustic model. Set this according to Kaldi's decision tree
para.topology.hiddenLayerSizeMask = [512];     % size of hidden layers in mask subnet
para.topology.hiddenLayerSizeBF = [];           % size of beamforming subnet if it's a neural networks. If it is MVDR, do not set it. 
para.topology.hiddenLayerSizeAM = [2048];       % size of acoustic model subnet.

% finish the rest of the configurations
para = ConfigMaskBFnetCE(para);
% generate the network
[Data_small, para] = LoadWavMask_Simu_CHiME4(para, 400, 'train'); % load a small amount of data for initialization purpose
[layer, para] = Build_MaskBFnet_CE(Data_small, para, 1);

% load the training and cv data
[Data_tr] = LoadWavMask_Simu_CHiME4(para, 100, 'train');
[Data_cv] = LoadWavMask_Simu_CHiME4(para, 100, 'dev');

% generate directory and file names to store the networks. 
para.output = sprintf('nnet/%s_Mask%dchOfficial.U%d.%d', para.topology.MaskNetType, para.topology.nChMask, length(Data_tr(1).data),  para.topology.nChMask*3*(para.topology.fft_len/2+1));
for i=1:length(para.topology.hiddenLayerSizeMask)
    para.output = sprintf('%s-%d', para.output, para.topology.hiddenLayerSizeMask(i));
end
para.output = sprintf('%s-%d.L2_%s.LR_%s/nnet', para.output, para.topology.fft_len/2+1, FormatFloat4Name(para.NET.L2weight),FormatFloat4Name(para.NET.learning_rate));
LOG = [];

% show the configurations
para
para.NET
para.IO
para.topology

% train the network using SGD
trainGraph_SGD(layer, Data_tr, Data_cv, para, LOG);