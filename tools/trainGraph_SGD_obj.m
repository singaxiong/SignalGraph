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
% Date Created: 10 Oct 2013
% Last Modified: 24 Oct 2015
%
% Updates in Version 12: 1) Support temporal convolution; 2) support
% minibatch that contains multiple trajectories of the same length (for
% temporal convolusion or LSTM); 3) Support dynamically generating training
% samples (pairs or noisy speech); 4) an enhanced FeaturePipe that can
% perform signal processing (such as STFT) and data manipulation
% (segmentation); 5) an enhanced stream synchronizer that can apply VAD on
% certain streams, and then extract fixed-length segments for training.
%
% Updates in Version 11: 1) modified the para structure. ParseOption2 now
% can output detailed descriptions of all parameters and set them to
% default values. 2) support base memory for Data to support efficient
% memory usage for pairwise training. 3) Finish the support for LSTM.
%
% Updates in Version 10: 1) support asynchronous input streams that have
% different frame rates. Now it is not necessary to convert the streams to
% uniform frame rates. 2) support LSTM; 3) support pairwise inputs and
% parameter tying, where the cost function depends on the distance or
% similarity between the input pairs. 
%
% Updates in Version 9: 1) support multi task learning that use the same
% inputs. 2) Support gated connection weighted_average; 
%
function trainGraph_SGD_obj(layer, data, data_t, para, LOG)
para = ParseOptions_obj(para);
layer = initializeParametersDNN_tree(layer, para);    % Use random initialization if a weight matrix is not defined.
layer = setDNNParameterPrecision(layer, para.singlePrecision, para.useGPU);
LOG = initializeLog(LOG);
para.NET.WeightUpdateOrder = genWeightUpdateOrder(layer, para.NET.WeightTyingSet);
layer = DetermineGradientPass(layer);

if para.IO.asyncFileRead
    WorkerPool = gcp('nocreate');   % create a background worker if it does not exist
    if isempty(WorkerPool)
        WorkerPool = parpool(1);
    end
else
    WorkerPool = [];
end

[filepath] = fileparts(para.output);    mkdir(filepath);

% --------------- Test on crossvalidation data before training -------- %
if para.skipInitialEval==0
    fprintf('Evaluating on cross-validation data - %s\n', datestr(now));
    [LOG.cost_cv0, cost_pure_cv, LOG.subcost_cv0, LOG.subacc_cv0] = CrossValidationTest_obj(layer, data_t, para, WorkerPool);
end

% --------------- START THE TRAINING --------------- %

startItr = length(LOG.actual_LR)+1;

% initialize momentum
weight_update_order = genWeightUpdateOrder(layer, para.NET.WeightTyingSet);
for k=1:length(weight_update_order)
    update{k}.W = 0;
    update{k}.b = 0;
end

nHourSeen = 0;
nHourSeen_reduce_lr = 0.5;       % this equals to about 0.5 hours of speech features if frame rate is 100Hz
next_Milestone = nHourSeen_reduce_lr*1;

if isfield(para, 'saveModelEveryXhours')
    next_Milestong_save = para.saveModelEveryXhours;
end

if ~isempty(LOG.actual_LR) && LOG.actual_LR(end)<para.NET.learning_rate
    learning_rate = LOG.actual_LR(end);
else
    learning_rate = para.NET.learning_rate;
end
para.NET.sentenceMinibatch = para.NET.sequential;
for itr = startItr:para.maxItr
    learning_rate = max(learning_rate, para.NET.learning_rate_floor);

    pause(1);

    % generate the data for current iteration. In normal mode, we do
    % nothing and use the same data for every iteration. If the mode is
    % dynamicSimulation, we will generate distorted speech again for each
    % iteration. 
    data = data.GenIterationData();
    
    % shuffle the training data, separate them into data blocks. One block
    % will be read into the memory. 
    data = data.ShuffleData(para);
    
    LOG.actual_LR(end+1) = learning_rate;
    layer_old = layer;  % store the old model
    
    cost_train = [];
    if para.useGPU; 	cost_train= gpuArray(cost_train);        end
    cost_train_pure = cost_train; subcost = cost_train;    subacc = cost_train;
    
    for blk_i = 1:data.nBlock
        pause(.1);
        
        if isfield(para, 'saveModelEveryXhours') && para.saveModelEveryXhours > 0
            if nHourSeen >=next_Milestong_save
                [~, cost_pure_cv] = CrossValidationTest_obj(layer, data_t, para, WorkerPool);
                modelfile = [para.output sprintf('.hours%d.LR%s', round(nHourSeen), FormatFloat4Name(learning_rate))];
                modelfile = sprintf('%s.CV%2.3f.mat', modelfile, cost_pure_cv(end));
                SaveDNN(layer, para, LOG, modelfile);                
                next_Milestong_save = next_Milestong_save + para.saveModelEveryXhours;
            end
        end
        if nHourSeen > next_Milestone
            learning_rate     = learning_rate * para.NET.learning_rate_decay_rate;   % Learning rate for biases of hidden units
            fprintf('Trained with %2.2f hours of data, reducing learning rate to %f\n', nHourSeen, learning_rate);
            next_Milestone = next_Milestone + nHourSeen_reduce_lr;
        end
        
        if para.useGPU && para.displayGPUstatus==1
            gpuStatus = gpuDevice;
            fprintf('GPU has %2.2E free memory!\n', gpuStatus.FreeMemory);
        end
        
        % ------------- prepare data for current block ------------- %
        
        % prepare minibatches
        if blk_i == 1 || ~para.IO.asyncFileRead
            minibatch = data.PrepareMinibatch(para.precision, para.NET.sequential, para.NET.batchSize, blk_i);
        else
            minibatch = fetchOutputs(MinibatchJob);     % collect the data from the work. It will block the main thread if the worker hasn't finished. 
        end
        if para.IO.asyncFileRead && blk_i<data.nBlock   % call a worker to load the next block of data in background
            MinibatchJob = parfeval(WorkerPool,@data.PrepareMinibatch,1, para.precision, para.NET.sequential, para.NET.batchSize, blk_i+1);
        end
        
        % ----------- train the network on current block ------------- %
        
        nMiniBatch = size(minibatch,2);
        cost_func = InitCostFunc(nMiniBatch, para);
        for batch_i=1:nMiniBatch
            batch_data = GetMinibatch(minibatch, batch_i, para.useGPU);
            
            % --------------- optional gradient check -------------------- %
            % ------------- Check whether the gradient is correct ------------- %
            if para.checkGradient ==1
                computeNumericalGradientLayer_tree2(layer, batch_data, para);
                para.checkGradient = 0; % do it only once
            end
                        
            nHourSeen = nHourSeen + size(batch_data{1},2)*size(batch_data{1},3) / data.streams(1).frameRate / 3600;
            % Evaluate the cost function and gradient on current batch
            % fprintf('Minibatch size = %d\n', size(batch_data{2},2))
            [cost_func_tmp,layer] = DNN_Cost10(layer, batch_data, para, 1);
            cost_func.cost(batch_i) = cost_func_tmp.cost;
            cost_func.cost_pure(batch_i) = cost_func_tmp.cost_pure;
            cost_func.subcost(:,batch_i) = cost_func_tmp.subcost(:);
            cost_func.subacc(:,batch_i) = cost_func_tmp.subacc(:);
            
            [layer, update, total_weight_norm] = DNN_update(layer, para, update, itr, learning_rate);
            
            if mod(batch_i,para.displayInterval)==0 || (batch_i==nMiniBatch && para.displayInterval>batch_i)
                fprintf('Cost at itr %i, blk %d/%d, MB %d/%d = %.4g / %.4g, W norm=%.4g', ...
                    itr, blk_i, data.nBlock, batch_i, nMiniBatch, ...
                    mean(cost_func.cost(max(1,batch_i-para.displayInterval+1) : batch_i)), ...
                    mean(cost_func.cost_pure(max(1,batch_i-para.displayInterval+1) : batch_i)), total_weight_norm);
                nCostFunc = length(para.cost_func.layer_idx);
                if nCostFunc>1
                    fprintf(' - Subcosts: ');
                    for cost_i = 1:nCostFunc
                        fprintf('%d) %f; ', cost_i, mean(cost_func.subcost(cost_i,max(1,batch_i-para.displayInterval+1) : batch_i)));
                    end
                end
                fprintf(' - %s - %s\n', datestr(now), para.displayTag);
                pause(0.01);
            end
            layer = clean_network_layer(layer);
        end
        cost_train = [cost_train [cost_func.cost]];
        cost_train_pure = [cost_train_pure [cost_func.cost_pure]];
        subcost = [subcost [cost_func.subcost]];
        subacc = [subacc [cost_func.subacc]];
        clear minibatch;
    end
    LOG.cost(itr) = gather(mean(cost_train));
    LOG.subcost(:,itr) = gather(mean(subcost,2));
    LOG.subacc(:,itr) = gather(mean(subacc,2));
    fprintf('Training cost/cost_pure at iteration %i = %f / %f\t - %s\n', ...
        itr, LOG.cost(itr), sum(LOG.subcost(:,itr)), datestr(now));
    if length(LOG.subcost(:,itr))>1
        fprintf('        ----subcosts = %f\n', LOG.subcost(:,itr));
    end
    if ~isempty(LOG.subacc(:,itr))
        fprintf('    ----(Sub)task accuracy = %2.2f%%\n', 100*LOG.subacc(:,itr));
    end
    
    % ----------- Evaluate on crossvalidation data ------------- %
    
    fprintf('Evaluating on cross-validation data - %s\n', datestr(now));
    [LOG.cost_cv(itr), cost_pure_cv, LOG.subcost_cv(:,itr), LOG.subacc_cv(:,itr)] = CrossValidationTest_obj(layer, data_t, para, WorkerPool);
    
    % ------------------ save current network ----------------------- %
    
    if isfield(para, 'saveModelEveryXIter') && mod(itr,para.saveModelEveryXIter)==0
        modelfile = [para.output sprintf('.itr%d.LR%s', itr, FormatFloat4Name(learning_rate))];
        if cost_pure_cv(end)>0.1
            modelfile = sprintf('%s.CV%2.3f.mat', modelfile, cost_pure_cv(end));
        else
            modelfile = sprintf('%s.CV%s.mat', modelfile, FormatFloat4Name(cost_pure_cv(end)));
        end            
        SaveDNN(layer, para, LOG, modelfile);
    end
    
    % ------------- stopping criterion and learn rate reduction --------- %
    
    switch para.NET.learningScheme
        case 'decayIfNoImprovement'     % if CV cost does not decrease, reduce the learning rate
            if itr>1 || para.NET.start_learning_rate_reduction==1
                if itr>=para.minItr && (LOG.subcost_cv(end-1)-LOG.subcost_cv(end))/LOG.subcost_cv(end) <para.NET.stopImprovement/100
                    fprintf('Improvement is less than %2.3f%%, stop the training!\n', para.NET.stopImprovement);
                    break;
                end
                if para.NET.start_learning_rate_reduction ==1 || (LOG.subcost_cv(end-1)-LOG.subcost_cv(end))/LOG.subcost_cv(end) <para.NET.reduceLearnRate/100
                    para.NET.start_learning_rate_reduction=1;
                    learning_rate = learning_rate * para.NET.reduceLearnRateSpeed;
                    fprintf('Cost reduction less than %2.3f%%, learning rate reduced by %2.2f%% to %f\n', para.NET.reduceLearnRate, 100-100*para.NET.reduceLearnRateSpeed, learning_rate);
                end
                if itr>startItr && sum(LOG.subcost_cv(end) > LOG.subcost_cv(1:end-1)) && para.NET.restore2prevModelIfFail % if the cost function increases, go back to the previous model.
                    layer = layer_old;
                    fprintf('Cost function increased in the latest iteration, restore to the previous best iteration network!\n');
                end
            end
            
        case 'expDecay'     % decay the learning rate no matter what happens to the CV cost. Run until a preset number of iterations is finished.
            learning_rate = learning_rate * para.NET.reduceLearnRateSpeed;
            if itr>=para.minItr
                break; end
    end
end
