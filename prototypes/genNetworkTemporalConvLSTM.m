% poolType = 'mean' or 'max'
% poolLayer = integer layer number after which we do the pooling
function layer = genNetworkTemporalConvLSTM(para)

layer = genNetworkTemporalConv2(para);
useFirstNLayer = length(para.nFilter)*3+1;
layer = layer(1:useFirstNLayer);

paraLSTM = para;
paraLSTM.inputDim = layer{end}.dim(1);
layerLSTM = genNetworkLSTM(paraLSTM);

layer = [layer layerLSTM(2:end)];

end

