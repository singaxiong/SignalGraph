% This is the OOP (object-oriented programming) version of the signal graph
% training.
%
function trainGraph_SGD_obj2(sg, data, data_t, learner, para, LOG)

LOG = LOG.initialize();

if para.IO.asyncFileRead
    WorkerPool = gcp('nocreate');   % create a background worker if it does not exist
    if isempty(WorkerPool)
        WorkerPool = parpool(1);
    end
else
    WorkerPool = [];
end

[filepath] = fileparts(para.output);    mkdir(filepath);

if para.skipInitialEval==0      % Test on crossvalidation data before training
    LOG = CrossValidationTest_obj2(sg, LOG, data, para, WorkerPool);
end

startItr = length(LOG.actual_LR)+1;
randomOrder = 1;

next_Milestong_save = para.saveModelEveryXhours;

for itr = startItr:para.maxItr
%     learning_rate = max(learning_rate, para.NET.learning_rate_floor);
    
    pause(1);
    
    % generate the data for current iteration. In normal mode, we do
    % nothing and use the same data for every iteration. If the mode is
    % dynamicSimulation, we will generate distorted speech again for each
    % iteration.
    data = data.GenIterationData();
    
    % shuffle the training data, separate them into data blocks. One block
    % will be read into the memory.
    data = data.ShuffleData(para.IO.maxNumSentInBlock);
    
    LOG.actual_LR(end+1) = learner.learningRate;
    LOG = LOG.initializeBlockCost();

    for blk_i = 1:data.nBlock
        pause(.1);
        
        if isfield(para, 'saveModelEveryXhours') && para.saveModelEveryXhours > 0
            if LOG.nHourSeen >=next_Milestong_save
                LOG_cv = CrossValidationTest_obj2(sg, data_t, para, WorkerPool);
                LOG.cost_cv_hour = LOG_cv.cost.appendCost(LOG.cost_cv_hour);
                
                SaveNet(itr, 1, sg, learner, LOG, para);
                next_Milestong_save = next_Milestong_save + para.saveModelEveryXhours;
            end
        end
                
        if para.useGPU && para.displayGPUstatus==1
            gpuStatus = gpuDevice; fprintf('GPU has %2.2E free memory!\n', gpuStatus.FreeMemory);
        end
               
        % prepare minibatches
        if blk_i == 1 || ~para.IO.asyncFileRead
            minibatch = data.PrepareMinibatch(blk_i, randomOrder);
        else
            minibatch = fetchOutputs(MinibatchJob);     % collect the data from the work. It will block the main thread if the worker hasn't finished.
        end
        if para.IO.asyncFileRead && blk_i<data.nBlock   % call a worker to load the next block of data in background
            MinibatchJob = parfeval(WorkerPool,@data.PrepareMinibatch,1, blk_i+1, randomOrder);
        end
        
        nMiniBatch = size(minibatch,2);
        LOG = LOG.initializeMiniBatchCost(length(sg.costLayerIdx), nMiniBatch, para.useGPU);

        for batch_i=1:nMiniBatch
            mb = data.GetOneMinibatch(minibatch, batch_i, para.useGPU);
            
            if para.checkGradient ==1   % optional gradient check, only do it once
                sg.verifyGradient(mb); para.checkGradient = 0; 
            end
                        
            sg = sg.forward(mb);
            cost = sg.evalCost();
            sg = sg.backward();
            sg = learner.updateGraph(sg);
            sg = sg.cleanUp();
            
            LOG = LOG.accumulateMiniBatchCost(batch_i, cost);
            LOG = LOG.accumulateTrainDataAmount(mb, data);
            LOG.DisplayCostMinibatch(itr, blk_i, data.nBlock, batch_i, nMiniBatch, para.displayInterval, para.displayTag);
            
            learner = learner.accumulateTrainDataDuration( size(mb{1},2)*size(mb{1},3) / data.streams(1).frameRate / 3600 );  % keep track of how much data we have observed
        end
        clear minibatch;
        LOG = LOG.accumulateBlockCost();
        learner = learner.updateLearningRateEndOfBlock();
    end
    learner = learner.updateLearningRateEndOfIteration();
    LOG = LOG.accumulateItrCost();
    LOG.DisplayCostIteration(itr);

    LOG_cv = CrossValidationTest_obj2(sg, data_t, para, WorkerPool);
    LOG.cost_cv = LOG_cv.cost.appendCost(LOG.cost_cv);

    SaveNet(itr, 0, sg, learner, LOG, para);
end
