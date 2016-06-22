function layer = genNetworkLSTM(para)
if isfield(para, 'LastActivation4MSE')==0
    para.LastActivation4MSE = 'linear';
end

layer{1}.name = 'Input';        % this is an input layer
layer{end}.inputIdx = 1;    % specifies the index of GCC in Visible_tr
layer{end}.dim = [1 1]*double(para.inputDim);             % [input dim; output dim];

if isfield(para, 'ProjectionSize') &&  ~isempty(para.ProjectionSize) && para.ProjectionSize>0
    layer{end+1}.name = 'Affine';
    layer{end}.prev = -1;
    layer{end}.W = [];
    layer{end}.b = [];
    layer{end}.dim = [para.ProjectionSize layer{end-1}.dim(1)];
    layer{end}.update = 1;
end

for i=1:length(para.hiddenLayerSizeLSTM)
    layer{end+1}.name = 'LSTM';
    layer{end}.prev = -1;   % this is the index of GCC features in Visible_tr; It is the offset to be added to current layer index
    layer{end}.W = []; % to be initialized randomly or by pretraining
    layer{end}.b = [];
    layer{end}.usePastState = para.usePastState(i);
    layer{end}.dim = [para.hiddenLayerSizeLSTM(i) layer{end-1}.dim(1)];
    layer{end}.update = 1;
end

layer2 = genNetworkFeedForward_v2(layer{end}.dim(1), para.hiddenLayerSizeFF, para.outputDim, para.costFn, para.LastActivation4MSE);

layer = [layer layer2(2:end)];

if isfield(para, 'labelDelay')
    layer{end}.labelDelay = para.labelDelay;
end
if isfield(para, 'costFrameSelection');
    layer{end}.costFrameSelection = para.costFrameSelection;
end

layer = FinishLayer(layer);
end

