function SaveNet(itr, useHourCost, sigGraph, learner, LOG, para)

if isfield(para, 'saveModelEveryXIter') && mod(itr,para.saveModelEveryXIter)~=0
    return;
end

modelfile = [para.output sprintf('.itr%d.hour%d.LR%s', itr, round(LOG.nHourSeen), FormatFloat4Name(learner.learningRate))];
if useHourCost
    cost_cv = LOG.cost_cv_hour.taskCost(end);
else
    cost_cv = LOG.cost_cv.taskCost(end);
end
if cost_cv>0.1
    modelfile = sprintf('%s.CV%2.3f.mat', modelfile, cost_cv);
else
    modelfile = sprintf('%s.CV%s.mat', modelfile, FormatFloat4Name(cost_cv));
end

sigGraph.useGPU = 0;
sigGraph = sigGraph.setPrecision();
sigGraph = sigGraph.cleanUp();
save(modelfile, 'sigGraph', 'para', 'learner', 'LOG');

end