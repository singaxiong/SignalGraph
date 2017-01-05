% This script train an LSTM based speech and noise mask predictor. The
% predicted masks will be used to derive MVDR beamforming filters which
% will be applied on the input data to obtain enhanced speech. The network
% will be trained to minimize the frame classification cross entropy, by
% using a DNN acoustic model. This script can use the initialized mask
% predictor trained in TrainMaskBF_MaskSubnet_CHiME4.m. 
% 
% Xiong Xiao, Nanyang Technological University, Singapore
% Last modified: 06 Dec 2016. 
%
clear
addpath('../lib','local');
para.IO.nStream = 2;        % two data streams, one is input signal, one is frame level triphone state label
para.NET.sequential = 1;    % do not randomize at frame level, randomize at sentence level
para.NET.variableLengthMinibatch = 1;   % a minibatch may contain multiple sentences of different lengths
para.NET.nSequencePerMinibatch = 10;     % number of sentences per minibatch
para.NET.maxNumSentInBlock = 100;       % maximum number of sentences in a block
para.NET.learning_rate = 3e-3;
para.NET.learning_rate_decay_rate = 0.995;
para.NET.start_learning_rate_reduction = 0;
para.NET.momentum = [0.9];
para.NET.L2weight = 0;
para.NET.gradientClipThreshold = 1;
para.NET.weight_clip = 10;
para.useGPU     = 1;                  % whether to use GPU
para.minItr     = 10;
para.displayGPUstatus = 1;
para.skipInitialEval = 0;
para.displayInterval = 3;
para.saveModelEveryXhours = 5;     % if the training is slow, we want to save the model between two iterations

% define local settings for the experiment, such as the data path.
chime_root = ChoosePath4OS({'F:/Data/CHiME4', '/home/xiaoxiong/CHiME4'});   % you can set two paths, first for windows OS and second for Linux OS. 
% ChoosePath4OS allows us to define two paths for the data, one for
% Windows system and one for Linux system. The function will select the
% correct path, so we don't need to change code for different platforms.
para.local.wavroot_noisy = [chime_root '/audio/isolated'];
para.local.wavroot_clean = [chime_root '/audio/isolated/tr05_org'];
% if you have at least 30GB free system memory, you can simply load all
% CHiME-4 waveforms (about 20GB) into memory. Otherwise, it is better to
% load the file names instead. Better to use SSD for fast loading speed.
para.local.useFileName = 1;      % if set to 0, load all training data to memory. otherwise, only load file names.
% define how to prepare the training data
para.local.data = 'mixed';  % choose the type of data for training: [mixed|simu|real]
para.local.pair = 'randPair';    % how do we generate channel pairs: [randPair|allPair]
para.local.aliDir = '../Kaldi/exp/tri3b_tr05_multi_noisy_ali';

% define network topology
para.topology.useWav = 1;       % use waveforms as input, perform STFT etc in the network
para.topology.useChannel = 6;   % use all 6 channels
para.topology.nChMask = 1;          % the number of channels we want to use as input of LSTM for mask prediction, usually set to 1, i.e. predict mask for each channel independently without using cross-channel information 
para.topology.MaskNetType = 'LSTM'; % define mask subnet type
para.topology.BfNetType = 'MVDR';   % define beamforming subnet type
para.topology.AmNetType = 'DNN';    % define acoustic model subnet type
para.topology.initialMask = 'nnet/LSTM_Mask1chOfficial.U41972.771-1024-257.L2_3E-4.LR_3E-2/nnet.itr7.LR7.93E-4.CV8.118.mat';    % the initial mask prediction network. If you don't provide the intial network, random initialization will be used. 
para.topology.initialAM_CE = '../Kaldi/exp-fbank/tri4a_dnn_tr05_multi_ch5';      % initial acoustic model, usually taking from Kaldi. If you don't provide the intial AM, random initialization will be used. 
para.topology.updateBF = 1;         % whether we want to update the mask prediction and beamforming network
para.topology.updateAM = 0;         % whether we want to update the acoustic model. 
para.topology.nSenone = 1981;       % number of senones, set the number according to the acoustic model. 
para.topology.hiddenLayerSizeMask = [1024];     % size of LSTM mask prediction network
para.topology.hiddenLayerSizeBF = [];           % size of beamforming network, don't set it as we are going to use MVDR to generate the beamforming filters. 
para.topology.hiddenLayerSizeAM = ones(1,7)*2048;   % size of acoustic model, set to 7x2048, the same as the CHiME-4 baseline system. 
para.topology.splitMask = 1;        % set to 1 if we want to predict speech and noise masks independently. 
para.topology.untieLSTM = 0;        % set to 1 if we don't want to share the LSTM between speech and noise mask prediction. 
para.topology.poolingType= 'none';  % type of pooling of masks of channels. 

% set a tag that will be displayed during training so we will know roughly
% what is being trained. 
para.displayTag = ['MaskBF_CE' num2str(para.topology.hiddenLayerSizeMask) '.updateAM' num2str(para.topology.updateAM)];

% finish the rest of the configuration
para = ConfigMaskBFnetCE(para);
% generate the network
[Data_small, para] = LoadWavLabel_CHiME4(para, 50, 'tr05');      % load a small amount of data for initialization purpose
% Build the basic network type, which does not use split mask prediction
% and pooling. 
[layer, para] = Build_MaskBFnet_CE(Data_small, para, 3);

para.output = sprintf('nnet/MaskBF%dch', para.topology.useChannel);
if para.topology.splitMask      % convert the basic model to splitMask if required
    [layer, para] = ConvertMaskBF2Split(layer, para);
    para.output = sprintf('%s_split%d', para.output, para.topology.untieLSTM);
end
if ~strcmpi(para.topology.poolingType, 'none')      % add pooling layers if required
    [layer, para] = ConvertMaskBF2pooling(layer, para);
    para.output = sprintf('%s_%s', para.output, para.topology.poolingType);
end

% load the training and dev data
[Data_tr, para] = LoadWavLabel_CHiME4(para, 1, 'tr05');
[Data_cv, para] = LoadWavLabel_CHiME4(para, 5, 'dt05');     % we only use 1/5 of dev data as cv data for speed. 

% generate the output directory name
para.output = sprintf('%s_%s_%s_%s.U%d_%s_%s.%d', para.output, para.topology.MaskNetType, para.topology.BfNetType, para.topology.AmNetType, ...
    length(Data_tr(1).data), para.local.data, para.local.pair, 3*(para.topology.fft_len/2+1));
for i=1:length(para.topology.hiddenLayerSizeMask)
    para.output = sprintf('%s-%d', para.output, para.topology.hiddenLayerSizeMask(i));
end
para.output = sprintf('%s-AM%d-%d_%d', para.output, para.topology.updateAM, length(para.topology.hiddenLayerSizeAM), para.topology.hiddenLayerSizeAM(1));
para.output = sprintf('%s-%d.L2_%s.LR_%s/nnet', para.output, para.topology.nSenone, FormatFloat4Name(para.NET.L2weight),FormatFloat4Name(para.NET.learning_rate));
LOG = [];

para
trainGraph_SGD(layer, Data_tr, Data_cv, para, LOG);

