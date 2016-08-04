% This script is a demonstration of how to train a toy acoustic model for
% English using filterbank features for the TIMIT task. Note that the demo
% does not contain the TIMIT corpus. You need to provide the TIMIT corpus
% root directory. 
%
% Warning: this demo assumes that your system has at least 8GB of free
% memory for Matlab.
%
% Author: Xiong Xiao, NTU, Singapore
% Date Created: 10 Oct 2013
% Last Modified: 16 Jun 2015
%

clear
para.IO.nStream = 2;        % Two input streams, one is the MFCC features, and the other is the frame-level phone label.
para.NET.sequential = 1;    % LSTM requires segment or setence based training
para.NET.batchSize = 256;   % this setting won't be used by LSTM
para.NET.nSequencePerMinibatch = 10;    % number of sentences in one minibatch
para.NET.variableLengthMinibatch = 1;   % whether the sentences in the minibatch are of different length
para.NET.maxNumSentInBlock = 100;       % maximum number of sentences in one block
para.NET.learning_rate = 1e-2;          % global learning rate
para.NET.learning_rate_decay_rate = 0.999;   % exponentially decay learning rate after every 0.5 hour (180000 frames)
para.NET.momentum = [0.5];          % momentun
para.NET.L2weight = 1e-4;           % L2 regularization weight
para.useGPU = 0;                    % whether to use GPU
para.displayInterval = 1;           % display training progress after N minibathces
para.checkGradient = 0;             % whether to perform gradient checking before training
para.stopImprovement = 0.1;         % when to stop the training, Most of time I use Ctr+C :)
para.reduceLearnRate = 0.5;         % when to start reducing learning rate
para.reduceLearnRateSpeed = 0.7;    % how fast do we decay the learning rate in each iteration
para.maxItr = 50;                   % maximum number of iterations allowed
para.minItr = 20;                   % maximum number of iterations allowed
para.NET.gradientClipThreshold = 1; % it's quite important to clip graidnet for LSTM training
para.NET.weight_clip = 10;          % limit the biggest weights

para.local.V = 39;                  % put task specific configurations into para.local. Here we specify that we use only 39 phones. 
para.local.fs = 16000;              % sampling rate, we can use 16000Hz or 8000Hz. 
para.local.doCMN = 1;               % whether to perform cepstral mean normalization for each utterance

[Data_tr, Data_cv, para] = LoadData_TIMIT(para, 'train');
para.IO.isTensor = [1 1];
para.IO.context = [1 1];    % we don't splice features when using LSTM

% generate the preprocessing of streams. We need to generate dynamic
% featuers for stream 1, i.e. the fbank stream. 
para.preprocessing{1}{1}.name = 'delta';
para.preprocessing{1}{end}.delta_order = 2;   % use upto second derivative of MFCCs
para.preprocessing{1}{end}.inputDim = 40;
para.preprocessing{1}{end}.outputDim = 120;

feat_tmp = cell2mat(Data_tr(1).data(1:10:end));
feat_tmp_delta = FeaturePipe(feat_tmp, para.preprocessing{1});  % generate the MFCCs with dynamic features
tmpProcessing = genDNNPreprocessing_splice_norm(feat_tmp_delta, para.IO.context(1));    % generate the splicing setting, and global mean and variance normalization processing
para.preprocessing{1} = [para.preprocessing{1} tmpProcessing];
feat_tmp2 = FeaturePipe(feat_tmp, para.preprocessing{1});
plot(mean(feat_tmp2')); hold on         % verify that we now have normal distributed input features
plot(std(feat_tmp2')); hold off;

para.preprocessing{2} = {};     % we don't need any preprocessing for the second stream, i.e. the label

para.topology.inputDim = para.preprocessing{1}{end}.outputDim;
para.topology.hiddenLayerSizeLSTM = [1024];              % number of cells, note that deep LSTM is not well tested yet
para.topology.usePastState = zeros(1,length(para.topology.hiddenLayerSizeLSTM)); % do not use peeping hole
para.topology.hiddenLayerSizeFF = [];               % hidden layers after LSTM
para.topology.outputDim = length(unique(cell2mat(Data_tr(2).data)));
para.topology.costFn = 'cross_entropy';
layer = genNetworkLSTM(para.topology);
para.cost_func.layer_idx = length(layer);
para.cost_func.layer_weight = [1];

para.output = sprintf('nnet/LSTM_phone_TIMIT');
para.output = sprintf('%s.U%d.%d', para.output, length(Data_tr(1).data), para.topology.inputDim);
for i=1:length(para.topology.hiddenLayerSizeLSTM)
    para.output = sprintf('%s-%d', para.output, para.topology.hiddenLayerSizeLSTM(i));
end
para.output = sprintf('%s-%d.L2_%s.LR_%s/nnet', para.output, para.topology.outputDim, FormatFloat4Name(para.NET.L2weight),FormatFloat4Name(para.NET.learning_rate));
LOG = [];

trainGraph_SGD(layer, Data_tr, Data_cv, para, LOG);



