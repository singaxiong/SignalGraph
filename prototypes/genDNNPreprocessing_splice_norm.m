function [preprocessing] = genDNNPreprocessing_splice_norm(visible, context, stage)
if nargin<3
    stage = 3;
end

if length(context)==2
    context_skip = context(2);
    context = context(1);
else
    context_skip = 0;
end

preprocessing{1}.name = 'splice';
half_ctx = (context-1)/2;
preprocessing{1}.transform = -half_ctx*(1+context_skip) : (1+context_skip) : half_ctx*(1+context_skip);
preprocessing{1}.inputDim = size(visible,1);
preprocessing{1}.outputDim = preprocessing{1}.inputDim * context;
if stage==1
    return;
end

mapped = FeaturePipe(visible, preprocessing);
preprocessing{2}.name = 'addshift';
preprocessing{2}.transform = mean(mapped');
preprocessing{2}.inputDim = preprocessing{1}.outputDim;
preprocessing{2}.outputDim = preprocessing{2}.inputDim;
if stage==2
    return;
end

preprocessing{3}.name = 'rescale';
preprocessing{3}.transform = 1./std(mapped');
preprocessing{3}.inputDim = preprocessing{2}.outputDim;
preprocessing{3}.outputDim = preprocessing{3}.inputDim;