% This script train an LSTM or DNN based clean speech log spectrogram 
% predictor, using simulated data of Reverb challenge. 
% 
% Xiong Xiao, Nanyang Technological University, Singapore
% Last modified: 08 Feb 2016. 
%
%clear
function TrainDereverbNet_Masking(modelType, hiddenLayerSize, hiddenLayerSizeFF, DeltaGenerationType, learning_rate,  ...
    step, nSequencePerMinibatch, useFileName, useGPU)
if nargin<3,    hiddenLayerSizeFF = [];     end
if nargin<4,    step = 1;   end
% if nargin<5,    useGPU = 1;     end

addpath('local', '../beamforming/lib');     % remember to run addMyPath.m in SignalGraph root directory first. 
% define SignalGraph settings. Type ParseOptions2() in command line to see the options. 
para.IO.nStream = 2;        % two data streams, one is input signal, one is clean target log spectrogram
para.NET.sequential = 1;    % do not randomize at frame level, randomize at sentence level
para.NET.variableLengthMinibatch = 1;   % a minibatch may contain multiple sentences of different lengths
para.NET.nSequencePerMinibatch = nSequencePerMinibatch;    % number of sentences per minibatch
para.NET.maxNumSentInBlock = 100;       % maximum number of sentences in a block
para.NET.L2weight = 3e-4;
para.NET.learning_rate = learning_rate;
para.NET.momentum = [0.9];
para.NET.learning_rate_decay_rate = 0.999;
para.NET.reduceLearnRateSpeed = 0.7;
para.NET.gradientClipThreshold = 1;
para.NET.weight_clip = 10;
para.useGPU     = useGPU;                  % don't use GPU for single sentence minibatch for LSTM. It's even slower than CPU. 
para.minItr     = 10;
para.displayGPUstatus = 1;
para.displayInterval = ceil(para.NET.maxNumSentInBlock / para.NET.nSequencePerMinibatch / 10);
para.skipInitialEval = 1;

reverb_root = ChoosePath4OS({'D:/Data/REVERB_Challenge', '/media/xiaoxiong/OS/data1/G/REVERB_Challenge'});   % you can set two paths, first for windows OS and second for Linux OS. 
para.local.wavroot = [reverb_root];
para.local.wsjcam0root = ChoosePath4OS({'D:/Data/wsjcam0', '/media/xiaoxiong/OS/data1/G/wsjcam0_wav'});
para.local.wsjcam0ext = ChoosePath4OS({'wav', 'wv1'});
para.local.useFileName = useFileName;      % if set to 0, load all training data to memory. otherwise, only load file names.
para.local.loadLabel = 0;
para.local.seglen = 100;
para.local.segshift = 100;

para.topology.useChannel = 1;
para.topology.useWav = 1;
para.topology.MaskNetType = modelType;
para.topology.useCMN = 0;
para.topology.DeltaGenerationType = DeltaGenerationType;   % Choose 'DeltaByEqn' or 'DeltaByAffine'
para.topology.MSECostWeightSDA = [1 4.5 10];
para.topology.hiddenLayerSize = hiddenLayerSize;
para.topology.hiddenLayerSizeFF = hiddenLayerSizeFF;
para = ConfigDereverbNet_Regression(para);

[Data_small, para] = LoadParallelWavLabel_Reverb(para, 100, 'train', {'simu'}, {'far'});
[layer, para] = Build_DereverbNet_Masking(Data_small, para);

% load the training and cv data
[Data_tr, para] = LoadParallelWavLabel_Reverb(para, step, 'train', {'simu'}, {'far'});
[Data_cv, para] = LoadParallelWavLabel_Reverb(para, step, 'dev', {'simu'}, {'far', 'near'});


% generate directory and file names to store the networks. 
if para.topology.useCMN; para.output = 'nnet/DereverbMask.CMN';
else;     para.output = 'nnet/DereverbMask.noCMN';    end
para.output = sprintf('%s.%s.MbSize%d.U%d.%d-%s', para.output, para.topology.DeltaGenerationType, ...
    para.NET.nSequencePerMinibatch, length(Data_tr(1).data),  3*(para.topology.fft_len/2+1), para.topology.RegressionNetType);
for i=1:length(para.topology.hiddenLayerSize)
    para.output = sprintf('%s-%d', para.output, para.topology.hiddenLayerSize(i));
end
if ~isempty(para.topology.hiddenLayerSizeFF)
    para.output = [para.output '-DNN'];
    for i=1:length(para.topology.hiddenLayerSizeFF)
        para.output = sprintf('%s-%d', para.output, para.topology.hiddenLayerSizeFF(i));
    end
end
para.output = sprintf('%s-%d.L2_%s.LR_%s/nnet', para.output, 3*(para.topology.fft_len/2+1), FormatFloat4Name(para.NET.L2weight),FormatFloat4Name(para.NET.learning_rate));
LOG = [];
para.displayTag = para.output(1:min(80, length(para.output)));

% show the configurations
para
para.NET
para.IO
para.topology

% train the network using SGD
trainGraph_SGD(layer, Data_tr, Data_cv, para, LOG);
