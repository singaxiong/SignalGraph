% This script is a demonstration of how to train a nonlinear mapping from
% noisy MFCC features to clean MFCC features. It can also be used for other
% types of regression tasks. 
%
% Author: Xiong Xiao, NTU, Singapore
% Date Created: 16 Jun 2016
% Last Modified: 16 Jun 2016
%

clear
para.IO.nStream = 2;        % Two input streams, one is the noisy MFCC features, and the other is the clean MFCC features
para.NET.sequential = 0;    % Use frame-based training, not sentence based training
para.NET.batchSize = 256;   % minibatch size
para.NET.learning_rate = 1e-2;      % global learning rate
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

[Data_tr, Data_cv, para] = LoadData_AU4_Mapping(para);

% generate the preprocessing of streams. 

feat_tmp = cell2mat(Data_tr(1).data(1:10:end));
para.preprocessing{1} = genDNNPreprocessing_splice_norm(feat_tmp, para.IO.context(1));    % generate the splicing setting, and global mean and variance normalization processing
feat_tmp2 = FeaturePipe(feat_tmp, para.preprocessing{1});
plot(mean(feat_tmp2')); hold on         % verify that we now have normal distributed input features
plot(std(feat_tmp2')); hold off;

para.preprocessing{2} = {};     % we don't need any preprocessing for the second stream, i.e. the label

inputDim = para.preprocessing{1}{end}.outputDim;
hiddenLayerSize = [512];        % you can generate deeper network by using something like: hiddenLayerSize = [512 512 200 512]. Then it will generate 4 hidden layers. 
outputDim = size(feat_tmp,1);
cost_function = 'mse';
layer = genNetworkFeedForward_v2(inputDim, hiddenLayerSize, outputDim, cost_function);
para.cost_func.layer_idx = length(layer);
para.cost_func.layer_weight = [1];

para.output = sprintf('nnet/DNN_mapping.U%d.%d', length(Data_tr(1).data), inputDim);
for i=1:length(hiddenLayerSize)
    para.output = sprintf('%s-%d', para.output, hiddenLayerSize(i));
end
para.output = sprintf('%s-%d.L2_%s.LR_%s/nnet', para.output, outputDim, FormatFloat4Name(para.NET.L2weight),FormatFloat4Name(para.NET.learning_rate));
LOG = [];

trainGraph_SGD(layer, Data_tr, Data_cv, para, LOG);



