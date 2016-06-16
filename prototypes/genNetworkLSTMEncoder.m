function layer = genNetworkLSTMEncoder(para)
if isfield(para, 'LastActivation4MSE')==0
    para.LastActivation4MSE = 'linear';
end

layer{1}.name = 'Input';        % this is an input layer
layer{end}.inputIdx = 1;    % specifies the index of GCC in Visible_tr
layer{end}.dim = [1 1]*double(para.inputDim);             % [input dim; output dim];

if isfield(para, 'useFbank') && para.useFbank
    MelWindow = mel_window_FE(40, 256, 16000)';
    MelWindow(:,end+1) = 0;
    layer{end+1}.name = 'Affine';
    layer{end}.prev = -1;
    layer{end}.W = MelWindow;
    layer{end}.b = zeros(40,1);
    layer{end}.dim = [40 layer{end-1}.dim(1)];
    layer{end}.update = 0;
    
    layer{end+1}.name = 'log';
    layer{end}.prev = -1;
    layer{end}.const = 1e-7;
    layer{end}.dim = [1 1]*layer{end-1}.dim(1);
end

layer{end+1}.name = 'delta';
layer{end}.prev = -1;
layer{end}.dim = [3 1]*layer{end-1}.dim(1);

layer{end+1}.name = 'Affine';
layer{end}.prev = -1;
layer{end}.W = [];
layer{end}.b = [];
layer{end}.dim = [1 1]*layer{end-1}.dim(1);
layer{end}.update = 0;

layerLSTM = genNetworkLSTM(para);
for i=1:length(layerLSTM)
    layerLSTM{i} = rmfield(layerLSTM{i}, 'next');
end

CostLayer = layerLSTM(end);
layerLSTM = layerLSTM(2:end-2);

layerLSTM{1}.prev = -1;
layerLSTM{1}.dim(2) = layer{end}.dim(1);

layer = [layer layerLSTM];

layer{end+1}.name = 'splice';
layer{end}.context = para.frames2encoder;
layer{end}.prev = 3-length(layer);
layer{end}.dim = [layer{end}.context 1]*layer{end+layer{end}.prev}.dim(1);

layer{end+1} = CostLayer{1};

if isfield(para, 'labelDelay')
    layer{end}.labelDelay = para.labelDelay;
end

layer = FinishLayer(layer);
end

