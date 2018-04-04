% This function do a cross validation test of DNN.
%
function LOG = CrossValidationTest_obj2(sigGraph, data, para, WorkerPool)
fprintf('Evaluating on cross-validation data - %s\n', datestr(now));
data = data.ShuffleData(para.IO.maxNumSentInBlock);

LOG = GraphTrainLog(sigGraph);
LOG = LOG.initializeBlockCost();
randomOrder = 0;

for blk_i = 1:data.nBlock
    pause(.1);
    
    if para.useGPU && para.displayGPUstatus==1
        gpuStatus = gpuDevice;
        fprintf('GPU has %2.2E free memory!\n', gpuStatus.FreeMemory);
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
    LOG = LOG.initializeMiniBatchCost(length(sigGraph.costLayerIdx), nMiniBatch, para.useGPU);
    
    for batch_i=1:nMiniBatch
        mb = data.GetOneMinibatch(minibatch, batch_i, para.useGPU);
        sigGraph = sigGraph.forward(mb);
        cost = sigGraph.evalCost();
        sigGraph = sigGraph.cleanUp();
        
        LOG = LOG.accumulateMiniBatchCost(batch_i, cost);
        LOG.DisplayCostMinibatch(0, blk_i, data.nBlock, batch_i, nMiniBatch, para.displayInterval, para.displayTag);
    end
    clear minibatch;
    LOG = LOG.accumulateBlockCost();
end
LOG = LOG.accumulateItrCost();
LOG.DisplayCostIteration(0);

end
