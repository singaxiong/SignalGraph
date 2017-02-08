% This script train an LSTM or DNN based clean speech log spectrogram 
% predictor, using simulated data of Reverb challenge. 
% 
% Xiong Xiao, Nanyang Technological University, Singapore
% Last modified: 08 Feb 2016. 
%
clear
addpath('local');     % remember to run addMyPath.m in SignalGraph root directory first. 
% define SignalGraph settings. Type ParseOptions2() in command line to see the options. 
para.IO.nStream = 2;        % two data streams, one is input signal, one is clean target log spectrogram
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
para.useGPU     = 1;                  % don't use GPU for single sentence minibatch for LSTM. It's even slower than CPU. 
para.minItr     = 10;
para.displayGPUstatus = 1;
para.displayInterval = 1;
para.skipInitialEval = 1;
para.displayTag = 'LSTM-Regression';

reverb_root = ChoosePath4OS({'D:/Data/REVERB_Challenge', '/home/xiaoxiong/CHiME4'});   % you can set two paths, first for windows OS and second for Linux OS. 
para.local.wavroot = [reverb_root];
para.local.useFileName = 1;      % if set to 0, load all training data to memory. otherwise, only load file names.

para.topology.useChannel = 1;
para.topology.RegressionNetType = 'LSTM';
para.topology.hiddenLayerSize = [1024];
para = ConfigDereverbNet_Regression(para);

dataset = 'eval';
datatype = 'simu';
distance = 'far';
[Data_small, para] = LoadParallelWavLabel_Reverb(para, 10, dataset, datatype, distance);
Data_small = Data_small([1 3]);
[layer, para] = Build_DereverbNet_Regression(Data_small, para);

% load the training and cv data
[Data_tr] = LoadWavMask_Simu_CHiME4(para, 1, 'train');
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