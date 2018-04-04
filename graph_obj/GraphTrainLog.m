classdef GraphTrainLog
    properties
        nCost = 1;
        cost; 
        cost_cv;
        cost_cv_hour;
        
        costMinibatch; 
        costBlock;
        
        nHourSeen = 0;
        actual_LR = [];
    end
    
    methods
        function obj = GraphTrainLog(sigGraph)
            obj.nCost = length(sigGraph.costWeight);
            obj.cost = GraphCost(obj.nCost, 0);
            obj.cost_cv = GraphCost(obj.nCost, 0);
            obj.cost_cv_hour = GraphCost(obj.nCost, 0);
            obj.costMinibatch = GraphCost(obj.nCost, 0);
            obj.costBlock = GraphCost(obj.nCost, 0);
        end
        
        function obj = initialize(obj)
            obj.actual_LR = [];
        end
        
        function obj = initializeBlockCost(obj)
            obj.costBlock = GraphCost(obj.nCost, 0);
        end
        
        function obj = initializeMiniBatchCost(obj,nCost, nBatch, useGPU)
            obj.costMinibatch = GraphCost(nCost, nBatch);
        end
        
        function obj = accumulateMiniBatchCost(obj, batch_i, cost_func)
            obj.costMinibatch = cost_func.copyCost(obj.costMinibatch, batch_i);
        end
        
        function obj = accumulateBlockCost(obj)
            obj.costBlock = obj.costMinibatch.appendCost(obj.costBlock);
        end
        
        function obj = accumulateItrCost(obj)
            obj.cost = obj.costBlock.appendMeanCost(obj.cost);
        end
        
        function obj = accumulateTrainDataAmount(obj, minibatch, data)
            for i=1:length(minibatch)
                [D,D2,T,N] = size(minibatch{i});
                if data.streams(i).frameRate==0
                    frameRate = 1;  % frameRate=0 means one data point per sequence. assume the sequence is 1s long. 
                else
                    frameRate = data.streams(i).frameRate;
                end
                duration(i) = T*N/frameRate;
            end
            obj.nHourSeen = obj.nHourSeen + max(duration) / 3600;
        end
        
        function SaveModel(obj, itr, para, sigGraph)
            if mod(itr,para.saveModelEveryXIter)==0
                modelfile = [para.output sprintf('.itr%d.LR%s', itr, FormatFloat4Name(learning_rate))];
                if obj.cost_pure_cv(end)>0.1
                    modelfile = sprintf('%s.CV%2.3f.mat', modelfile, cost_pure_cv(end));
                else
                    modelfile = sprintf('%s.CV%s.mat', modelfile, FormatFloat4Name(cost_pure_cv(end)));
                end
                LOG = obj;
                SaveDNN(sigGraph, para, LOG, modelfile);
            end
        end
        
        function DisplayCostMinibatch(obj, itr, blk_i, nBlock, batch_i, nMiniBatch, displayInterval, displayTag)
            if itr<=0
                ItrStr = 'CV cost'; 
            else
                ItrStr = sprintf('Cost at itr %i', itr);
            end
                
            if mod(batch_i,displayInterval)==0 || (batch_i==nMiniBatch && displayInterval>batch_i)
                costMB = obj.costMinibatch;
                meanTotalCost = mean(costMB.totalCost(max(1,batch_i-displayInterval+1) : batch_i));
                if isnan(meanTotalCost)
                    pause(0.1);
                end
                fprintf('%s, blk %d/%d, MB %d/%d = %.4g / %.4g', ...
                    ItrStr, blk_i, nBlock, batch_i, nMiniBatch, ...
                    meanTotalCost, ...
                    mean(costMB.taskCost(max(1,batch_i-displayInterval+1) : batch_i)));
                if obj.nCost>1
                    fprintf(' - Subcosts: ');
                    for i = 1:obj.nCost
                        fprintf('%d) %f; ', i, mean(costMB.subCost(i,max(1,batch_i-displayInterval+1) : batch_i)));
                    end
                end
                fprintf(' - %s - %s\n', datestr(now), displayTag);
                pause(0.01);
            end
        end
        
        function DisplayCostIteration(obj, itr)         
            if itr<=0
                ItrStr = 'CV cost'; 
            else
                ItrStr = sprintf('Cost at itr %i', itr);
            end
            
            fprintf('%s = %f / %f\t - %s\n', ...
                ItrStr, obj.cost.totalCost(end), sum(obj.cost.taskCost(:,end)), datestr(now));
            if length(obj.cost.subCost(:,end))>1
                fprintf('        ----subcosts = %f\n', obj.cost.subCost(:,end));
            end
            if sum(obj.cost.subAcc(:,end)>=0)
                fprintf('    ----(Sub)task accuracy = %2.2f%%\n', 100*obj.cost.subCost(:,end));
            end
        end
    end
    
    methods
        


    end
    
end