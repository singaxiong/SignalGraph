function [processing] = genDNNPreprocessing_splice_norm_tree(visible, reader, preprocessing, context, nUttUsed, stage)
if exist('nUttUsed')==0 || length(nUttUsed)==0
    nUttUsed = 500;
end
if exist('stage')==0 || length(stage)==0
    stage = 3;
end

nUtt = length(visible.data);
if nUtt>nUttUsed
    step = round(nUtt/nUttUsed);
    data = visible.data(1:step:end);
else
    data = visible.data;
end

if visible.isFileName
    for i=1:length(data)
        feat{i} = InputReader(data{i}, reader, 0, 0);
    end
else
    feat = data;
end

if length(preprocessing)>0
    for i=1:length(feat)
        feat{i} = FeaturePipe(feat{i}, preprocessing);
    end
end

feat = cell2mat(feat);

processing{1}.name = 'splice';
half_ctx = (context-1)/2;
processing{1}.transform = -half_ctx:half_ctx;
processing{1}.inputDim = size(feat,1);
processing{1}.outputDim = processing{1}.inputDim * context;
if stage==1
    return;
end

mapped = FeaturePipe(feat, processing);
processing{2}.name = 'addshift';
processing{2}.transform = mean(mapped');
processing{2}.inputDim = processing{1}.outputDim;
processing{2}.outputDim = processing{2}.inputDim;
if stage==2
    return;
end

processing{3}.name = 'rescale';
processing{3}.transform = 1./std(mapped');
processing{3}.inputDim = processing{2}.outputDim;
processing{3}.outputDim = processing{3}.inputDim;

mapped = FeaturePipe(feat, processing);
plot(mean(mapped,2)); hold on;
plot(std(mapped')); hold off

processing = [preprocessing processing];

end
