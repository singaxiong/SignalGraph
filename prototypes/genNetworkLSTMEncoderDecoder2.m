function layer = genNetworkLSTMEncoderDecoder2(para)
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

layerLSTM = layerLSTM(2:end);

layerLSTM{1}.prev = -1;
layerLSTM{1}.dim(2) = layer{end}.dim(1);

layer = [layer layerLSTM];

layer{end-2}.name = 'multi_softmax';
layer{end-2}.TaskVocabSizes = para.outputDim/para.ngram_len;
if isfield(para, 'labelDelay')
    shiftLayer.name = 'frame_shift';
    shiftLayer.prev = -1;
    shiftLayer.delay = para.labelDelay;
    
end


layer{end}.name = 'multi_cross_entropy';
layer{end}.TaskVocabSizes = para.outputDim/para.ngram_len;
layer = FinishLayer(layer);
end

