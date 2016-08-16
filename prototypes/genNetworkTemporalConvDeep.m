% poolType = 'mean' or 'max'
% poolLayer = integer layer number after which we do the pooling
function layer = genNetworkTemporalConvDeep(para)
if isfield(para, 'LastActivation4MSE')==0
    para.LastActivation4MSE = 'linear';
end
inputDim = double(para.inputDim);

layer{1}.name = 'Input';        % this is an input layer
layer{end}.inputIdx = 1;    % specifies the index of GCC in Visible_tr
layer{end}.dim = [1 1]*inputDim;             % [input dim; output dim];

nFilter = para.nFilter;
for i=1:length(nFilter)
    if i==1 % only the first layer will use tconv, the rest uses splice and affine transform
        layer{end+1}.name = 'tconv';
        layer{end}.prev = -1;
        layer{end}.W = []; % to be initialized randomly or by pretraining
        layer{end}.b = [];
        layer{end}.dim = [nFilter(i) inputDim*para.filterLen(i)];
    else
        layer{end+1}.name = 'Splice';
        layer{end}.prev = -1;
        layer{end}.context = para.filterLen(i);
        layer{end}.dim = [layer{end}.context 1]*layer{length(layer)+layer{end}.prev}.dim(1);
        layer{end}.update = 0;

        layer{end+1}.name = 'Affine';
        layer{end}.prev = -1;
        layer{end}.W = []; % to be initialized randomly or by pretraining
        layer{end}.b = [];
        layer{end}.dim = [nFilter(i) layer{length(layer)+layer{end}.prev}.dim(1)];
    end
    layer{end}.update = 1;
    
    layer{end+1}.name = 'tmaxpool';
    layer{end}.context = para.poolingLen(i);
    layer{end}.stride = para.poolingStride(i);
    layer{end}.prev = -1;
    layer{end}.dim = [1 1]*nFilter(i);
    
    if isfield(para, 'activation')
        layer{end+1}.name = para.activation;
    else
        layer{end+1}.name = 'sigmoid';
    end
    layer{end}.prev = -1;
    layer{end}.dim = [1 1]*nFilter(i);
end

layer2 = genNetworkFeedForward_v2(nFilter(end), para.hiddenLayerSizeFF, para.outputDim, para.costFn, para.LastActivation4MSE);

layer = [layer layer2(2:end)];
layer = FinishLayer(layer);

end

