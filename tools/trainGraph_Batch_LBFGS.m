% This function optimizes a computational network by using stochastic 
% gradient descent (SGD). The inputs:
%   layer - optional initialization of the parameters of DNN
%   visible - the input of the DNN of training samples
%   target - the desired output of the DNN of the training samples
%   visible_t - the input of the DNN of cross validation samples
%   target_t - the desired output of the DNN of the vross validation samples
%   para - a structure that contains the settings of DNN
%   LOG - a structure that records training process information for debugging
%
% Author: Xiong Xiao, Temasek labs, NTU, Singapore.
% Date Created: 10 Oct 2015
% Last Modified: 04 Jul 2016
%
%
function layer = trainGraph_Batch_LBFGS(layer, data, data_t, para, LOG)
para = ParseOptions2(para);
layer = initializeParametersDNN_tree(layer, para);    % Use random initialization if a weight matrix is not defined.
layer = setDNNParameterPrecision(layer, para.singlePrecision, para.useGPU);
LOG = initializeLog(LOG);
para.NET.WeightUpdateOrder = genWeightUpdateOrder(layer, para.NET.WeightTyingSet);

[filepath] = fileparts(para.output);    mkdir(filepath);

% --------------- Test on crossvalidation data before training -------- %
if ~isempty(data_t) && para.skipInitialEval==0
    fprintf('Evaluating on cross-validation data - %s\n', datestr(now));
    [LOG.cost_cv0, cost_pure_cv, LOG.subcost_cv0, LOG.subacc_cv0] = CrossValidationTest_tree4(layer, data_t, para);
end

% --------------- START THE TRAINING --------------- %

startItr = length(LOG.actual_LR)+1;

para.NET.sentenceMinibatch = para.NET.sequential;
[minibatch] = MinibatchPackaging_tree4(data, para);
batch_data = minibatch.data(:,1);

for itr = startItr:para.maxItr
    
    % --------------- optional gradient check -------------------- %
    % ------------- Check whether the gradient is correct ------------- %
    if para.checkGradient ==1
        computeNumericalGradientLayer_tree2(layer, batch_data, para);
        para.checkGradient = 0; % do it only once
    end
    
    options.maxIter = para.NET.maxItrLBFGS;
    options.Method = 'LBFGS';
    options.display = 1;
    
    mode = 1;
    initW = NetWeights_layer2vec(layer, 0, para.useGPU);
    [opttheta, cost(itr)] = minFunc( @(x) DNN_cost_wrapper(x, layer, batch_data, para, mode), initW, options); % optimization
    layer = NetWeights_vec2layer(opttheta, layer, 0);
    
    % ----------- Evaluate on crossvalidation data ------------- %    
    if ~isempty(data_t)
        fprintf('Evaluating on cross-validation data - %s\n', datestr(now));
        [LOG.cost_cv(itr), cost_pure_cv, LOG.subcost_cv(:,itr), LOG.subacc_cv(:,itr)] = CrossValidationTest_tree4(layer, data_t, para);
    end
end
