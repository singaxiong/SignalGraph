% This script train an LSTM or DNN based clean speech log spectrogram 
% predictor, using simulated data of Reverb challenge. 
% 
% Xiong Xiao, Nanyang Technological University, Singapore
% Last modified: 08 Feb 2016. 
%
%clear
function TrainEnhanceNet_Masking(modelType, hiddenLayerSize, hiddenLayerSizeFF, DeltaGenerationType, learning_rate,  ...
    nUtt4Iteration, nSequencePerMinibatch, useFileName, useGPU)

addpath('local', '../beamforming/lib', '../dereverb/local');     % remember to run addMyPath.m in SignalGraph root directory first. 
% define SignalGraph settings. Type ParseOptions2() in command line to see the options. 
para.IO.nStream = 2;        % two data streams, one is input signal, one is clean target log spectrogram
para.IO.mode = 'dynamicDistortion';     % distorted speech is generated dynamically during training. So each iteration uses different data. 
para.NET.sequential = 1;    % do not randomize at frame level, randomize at sentence level
para.NET.variableLengthMinibatch = 1;   % a minibatch may contain multiple sentences of different lengths
para.NET.nSequencePerMinibatch = nSequencePerMinibatch;    % number of sentences per minibatch
para.NET.nSequencePerMinibatchCV = nSequencePerMinibatch/5; % number of sentences per minibatch for CV
para.NET.maxNumSentInBlock = 100;       % maximum number of sentences in a block
para.NET.L2weight = 3e-4;
para.NET.learning_rate = learning_rate;
para.NET.momentum = [0.9];
para.NET.learning_rate_decay_rate = 0.999;
para.NET.reduceLearnRateSpeed = 0.9;
para.NET.gradientClipThreshold = 1;
para.NET.weight_clip = 3;
para.useGPU     = useGPU;                  % don't use GPU for single sentence minibatch for LSTM. It's even slower than CPU. 
para.minItr     = 100;
para.displayGPUstatus = 1;
para.displayInterval = ceil(para.NET.maxNumSentInBlock / para.NET.nSequencePerMinibatch / 10);
para.skipInitialEval = 1;

para.local.clean_wav_root = ChoosePath4OS({'D:\Data\NoiseData\Libri\LibriSpeech\train-clean-100', '/media/xiaoxiong/DATA1/data1/Libri/LibriSpeech/train-clean-100'}); 
para.local.clean_wav_ext = 'flac';
para.local.rir_wav_root = ChoosePath4OS({'D:\Data\NoiseData\ReverbEstimation\RIR_T60_1ch', '/media/xiaoxiong/DATA1/data1/ReverbEstimation/RIR_T60_1ch'}); 
para.local.rir_wav_ext = 'wav';
para.local.noise_wav_root = ChoosePath4OS({'D:\Data\NoiseData\musan\noise', '/media/xiaoxiong/DATA1/data1/musan/noise'}); 
para.local.noise_wav_ext = 'wav';
para.local.useFileName = useFileName;      % if set to 0, load all training data to memory. otherwise, only load file names.

para.topology.useChannel = 1;
para.topology.useWav = 1;
para.topology.MaskNetType = modelType;
para.topology.useCMN = 0;
para.topology.DeltaGenerationType = DeltaGenerationType;   % Choose 'DeltaByEqn' or 'DeltaByAffine'
para.topology.MSECostWeightSDA = [1 4.5 10];
para.topology.hiddenLayerSize = hiddenLayerSize;
para.topology.hiddenLayerSizeFF = hiddenLayerSizeFF;
para = ConfigDereverbNet_Regression(para);

para.IO.DynamicDistortion.fs = para.topology.fs;
para.IO.DynamicDistortion.nUtt4Iteration = nUtt4Iteration;   % dynamically generate 10000 distorted sentences for each training iteration
para.IO.DynamicDistortion.nHours4Iteration = 10;    % dynamically generate 10 hours of distorted speech for each training iteration
para.IO.DynamicDistortion.SNR_PDF = 'uniform';      % distribution of SNR, choose [uniform|normal]
para.IO.DynamicDistortion.SNR_para = [-20 30];      % parameters of SNR PDF. If use uniform, it is the lowest and highest SNR allowed. 
                                                    % if use normal, it is the mean and variance of SNR. 
para.IO.DynamicDistortion.seglen = 100;             % cut distorted sentences into equal length segments. 
para.IO.DynamicDistortion.segshift = 100;
para.IO.DynamicDistortion.frame_len = para.topology.frame_len;
para.IO.DynamicDistortion.frame_shift = para.topology.frame_shift;

[Data_tr, para] = LoadWavRIRNoise_Libri(para, 1);

[layer, para] = Build_EnhanceNet_Masking(Data_tr, para);

% load the cv data
para.local.cv_wav_root = ChoosePath4OS({'D:\Data\NoiseData\Libri\LibriSpeech\dev-parallel', '/media/xiaoxiong/DATA1/data1/Libri/LibriSpeech/dev-parallel'}); 
para.local.useFileName = 0;
[Data_cv, para] = LoadParallelWav_Libri(para, 2);

% generate directory and file names to store the networks. 
if para.topology.useCMN; para.output = 'nnet/EnhanceMask.CMN';
else;     para.output = 'nnet/EnhanceMask.noCMN';    end
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
para.displayTag = para.output(1:min(90, length(para.output)));

% show the configurations
para
para.NET
para.IO
para.topology

% train the network using SGD
trainGraph_SGD(layer, Data_tr, Data_cv, para, LOG);
