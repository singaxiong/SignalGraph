function pretrain_DNN(visible_train, visible_cv, para, nnet_folder)

if isfield(para, 'useGPU')==0;                  para.useGPU         = 0;            end         % whether to use GPU
% Learning control
if isfield(para, 'InputPDF')==0;                para.InputPDF  = 'Gaussian';        end         % initial learning rate
if isfield(para, 'learning_rate')==0;           para.learning_rate  = 0.01;         end         % initial learning rate
if isfield(para, 'batchSize')==0;               para.batchSize = 256;               end         % size of minibatch. After each minibatch, we update the DNN parameters. 
if isfield(para, 'stopImprovement')==0;         para.stopImprovement = 0.01;        end         % when to stop the training
if isfield(para, 'reduceLearnRate')==0;         para.reduceLearnRate = 0.2;         end         % when to start reducing learning rate
if isfield(para, 'reduceLearnRateSpeed')==0;    para.reduceLearnRateSpeed = 0.5;    end         % how fast do we decay the learning rate in each iteration
if isfield(para, 'saveModelEveryXIter')==0;     para.saveModelEveryXIter = 1;       end         % how often do we save the weights
if isfield(para, 'L2weight')==0;                para.L2weight = 3e-5;               end         % weight of L2 regularization, i.e. weight decay
if isfield(para, 'displayInterval')==0;         para.displayInterval = 100;         end         % display the cost of training every x minibatches
if isfield(para, 'checkGradient')==0;           para.checkGradient = 0;             end         % performing gradient checking before training
if isfield(para, 'momentum')==0;                para.momentum = [0.5 0.5];          end

% generate data preprocessing
for i=1:length(visible_train); nFr(i) = size(visible_train{i}, 2); end
step = ceil(sum(nFr)/5e4);
featTmp = cell2mat(visible_train(1:step:end));
para.preprocessing = genDNNPreprocessing_splice_norm(featTmp, para.context);
%featTmp2 = FeaturePipe(featTmp, para.preprocessing);

% decide network size
inputSize = para.preprocessing{end}.outputDim;
para.layerSize = [inputSize para.layerSize];

% define the output folder
if nargin<4
    nnet_folder = 'rbm/pretrain';
end
for i=1:length(para.layerSize)
    nnet_folder = sprintf('%s%d-', nnet_folder, para.layerSize(i));
end
nnet_folder(end) = [];
my_mkdir(nnet_folder);

for layer_i = 2:length(para.layerSize)
    para2 = para;
    para2.layerSize = para.layerSize(layer_i-1:layer_i);

    if layer_i==2 && strcmpi(para.InputPDF, 'Gaussian')
        para2.InputPDF = 'Gaussian';     % define the output node type
    else
        para2.InputPDF = 'Bernoulli';     % define the output node type
    end

    para2.maxItr = 200;                   % maximum number of iterations allowed
    para2.minItr = 30;                   % maximum number of iterations allowed
    para2.rbm = 1;
    
    if layer_i == 2
        
    else
        % search in the previous layer folder the best model to load
        sorted_model_files = sort_nnet_by_dev(findFiles([nnet_folder sprintf('/layer%d',layer_i-1)], 'mat'));
%         [preProcessing prev_model] = LoadDNNasFeaturePipe_v2(sorted_model_files{1}, para.useGPU);
%         para.preprocessing = preProcessing(1:end-2);
%         para.layerSize = [para.preprocessing{end-1}.outputDim para.hiddenLayerSize outputSize];
    end
    
    % decide network size
    para2.output = sprintf('%s/layer%d/rbm', nnet_folder, layer_i);
    [outpath] = fileparts(para2.output); my_mkdir(outpath);
        
    init_rbm = {};
    final_rbm = train_RBM_CD_v2(init_rbm, visible_train, visible_cv, para2);
end
end
