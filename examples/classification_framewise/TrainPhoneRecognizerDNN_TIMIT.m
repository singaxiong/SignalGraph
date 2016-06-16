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
para.NET.sequential = 0;    % Use frame-based training, not sentence based training
para.NET.batchSize = 256;   % minibatch size
para.NET.learning_rate = 3e-2;      % global learning rate
para.NET.momentum = [0.5];          % momentun
para.NET.L2weight = 3e-4;           % L2 regularization weight
para.useGPU = 0;                    % whether to use GPU
para.displayInterval = 100;         % display training progress after N minibathces
para.checkGradient = 0;             % whether to perform gradient checking before training
para.stopImprovement = 0.1;         % when to stop the training, Most of time I use Ctr+C :)
para.reduceLearnRate = 0.5;         % when to start reducing learning rate
para.reduceLearnRateSpeed = 0.7;    % how fast do we decay the learning rate in each iteration
para.maxItr = 50;                   % maximum number of iterations allowed
para.minItr = 10;                   % maximum number of iterations allowed

para.local.V = 39;                  % put task specific configurations into para.local. Here we specify that we use only 39 phones. 
para.local.fs = 16000;              % sampling rate, we can use 16000Hz or 8000Hz. 
para.local.doCMN = 1;               % whether to perform cepstral mean normalization for each utterance

[Data_tr, Data_cv, para] = LoadData_TIMIT(para, 'train');

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

inputDim = para.preprocessing{1}{end}.outputDim;
hiddenLayerSize = [1024 1024 1024];        % you can generate deeper network by using something like: hiddenLayerSize = [512 512 200 512]. Then it will generate 4 hidden layers. 
outputDim = length(unique(cell2mat(Data_tr(2).data)));
cost_function = 'cross_entropy';
layer = genNetworkFeedForward_v2(inputDim, hiddenLayerSize, outputDim, cost_function);
para.cost_func.layer_idx = length(layer);
para.cost_func.layer_weight = [1];

para.output = sprintf('nnet/DNN_phone_TIMIT');
para.output = sprintf('%s.U%d.%d', para.output, length(Data_tr(1).data), inputDim);
for i=1:length(hiddenLayerSize)
    para.output = sprintf('%s-%d', para.output, hiddenLayerSize(i));
end
para.output = sprintf('%s-%d.L2_%s.LR_%s/nnet', para.output, outputDim, FormatFloat4Name(para.NET.L2weight),FormatFloat4Name(para.NET.learning_rate));
LOG = [];

trainGraph_SGD(layer, Data_tr, Data_cv, para, LOG);



