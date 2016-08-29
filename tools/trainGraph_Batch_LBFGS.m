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
% Last Modified: 07 Jul 2016
%
%
function [layer, para, LOG] = trainGraph_Batch_LBFGS(layer, data, data_t, para, LOG)
para = ParseOptions2(para);
layer = initializeParametersDNN_tree(layer, para);    % Use random initialization if a weight matrix is not defined.
layer = setDNNParameterPrecision(layer, para.singlePrecision, para.useGPU);
LOG = initializeLog(LOG);
para.NET.WeightUpdateOrder = genWeightUpdateOrder(layer, para.NET.WeightTyingSet);

% [filepath] = fileparts(para.output);    mkdir(filepath);

% --------------- Test on crossvalidation data before training -------- %
if ~isempty(data_t) && para.skipInitialEval==0
    fprintf('Evaluating on cross-validation data - %s\n', datestr(now));
    [LOG.cost_cv0, cost_pure_cv, LOG.subcost_cv0, LOG.subacc_cv0] = CrossValidationTest_tree4(layer, data_t, para);
end

% --------------- START THE TRAINING --------------- %

para.NET.sentenceMinibatch = para.NET.sequential;
[minibatch] = MinibatchPackaging_tree4(data, para);
batch_data = minibatch.data(:,1);

% ------------- Check whether the gradient is correct ------------- %
if para.checkGradient ==1
    computeNumericalGradientLayer_tree2(layer, batch_data, para);
end

options.maxIter = para.NET.maxItrLBFGS;
options.Method = 'LBFGS';
options.display = para.NET.displayLBFGS;    % disable the message from the minFunc package

mode = 1;
startItr = length(LOG.actual_LR)+1;
theta = NetWeights_layer2vec(layer, 0, para.useGPU);

[cost_func, layer] = DNN_Cost10(layer, batch_data, para, mode); 
LOG.cost0 = cost_func.cost;
for itr = startItr:para.maxItr
    old_theta = theta;
    
    [cost_func, layer] = DNN_Cost10(layer, batch_data, para, mode);     % run the DNN_Cost once to generate some parameters, such as posteriors of Gaussians
    
    [theta, cost] = minFunc( @(x) DNN_cost_wrapper(x, layer, batch_data, para, mode), theta, options); % optimization
    LOG.cost(itr) = gather(cost);

    layer = NetWeights_vec2layer(theta, layer, 0);
    
    if ~isfield(para, 'cleanLayer') || para.cleanLayer
        layer = clean_network_layer(layer);         % clean the network, such as posteriors of Gaussians
    end
    
    fprintf('Cost for epoch %d = %f - %s\n', itr, LOG.cost(itr), datestr(now));
    
    % ----------- Evaluate on crossvalidation data ------------- %    
    if ~isempty(data_t)
        fprintf('Evaluating on cross-validation data - %s\n', datestr(now));
        [LOG.cost_cv(itr), cost_pure_cv, LOG.subcost_cv(:,itr), LOG.subacc_cv(:,itr)] = CrossValidationTest_tree4(layer, data_t, para);
    end
    
    % stopping criterion
    if itr>1 && itr>=para.minItr && (LOG.cost(end-1)-LOG.cost(end))/LOG.cost(end) <para.NET.stopImprovement/100
        fprintf('Improvement is less than %2.3f%%, stop the training!\n', para.NET.stopImprovement);
        break;
    end
end
