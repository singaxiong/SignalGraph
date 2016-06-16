% poolType = 'mean' or 'max'
% poolLayer = integer layer number after which we do the pooling
function layer = genNetworkTemporalConv2(para)
if isfield(para, 'LastActivation4MSE')==0
    para.LastActivation4MSE = 'linear';
end
inputDim = double(para.inputDim);

layer{1}.name = 'Input';        % this is an input layer
layer{end}.inputIdx = 1;    % specifies the index of GCC in Visible_tr
layer{end}.dim = [1 1]*inputDim;             % [input dim; output dim];

nFilter = para.nFilter;
for i=1:length(nFilter)
    layer{end+1}.name = 'tconv';
    layer{end}.prev = -1;
    layer{end}.W = []; % to be initialized randomly or by pretraining
    layer{end}.b = [];
    if i==1
        layer{end}.dim = [nFilter(i) inputDim*para.filterLen(i)];
    else
        layer{end}.dim = [nFilter(i) nFilter(i-1)*para.filterLen(i)];
    end
    layer{end}.update = 1;
    
    layer{end+1}.name = 'tmaxpool';
    layer{end}.context = para.poolingLen(i);
    layer{end}.stride = para.poolingStride(i);
    layer{end}.prev = -1;
    layer{end}.dim = [1 1]*nFilter(i);
    

    layer{end+1}.name = 'sigmoid';
    layer{end}.prev = -1;
    layer{end}.dim = [1 1]*nFilter(i);
end

layer2 = genNetworkFeedForward_v2(nFilter(end), para.hiddenLayerSizeFF, para.outputDim, para.costFn, para.LastActivation4MSE);

layer = [layer layer2(2:end)];


% automatically derive the list of layers that the output of the current layer goes.
for i=1:length(layer); layer{i}.next = []; end
for i=length(layer):-1:1
    if isfield(layer{i}, 'prev')
        for j=1:length(layer{i}.prev)
            layer{i+layer{i}.prev(j)}.next(end+1) = -layer{i}.prev(j);
        end
    end
end

end

