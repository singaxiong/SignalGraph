function layer = genNetworkLSTMTopicNgram(para)
if isfield(para, 'LastActivation4MSE')==0
    para.LastActivation4MSE = 'linear';
end

vocabSize = para.outputDimNgram;
layer = genNetworkLSTM(para);
idx2vec_layer.name = 'idx2vec';
idx2vec_layer.dim = [vocabSize 1];
idx2vec_layer.prev = -1;
idx2vec_layer.next = 1;
layer = [layer(1) idx2vec_layer layer(2:end)];
layer{3}.name = 'word2vec';
layer{3}.context = 1;
layer{3} = rmfield(layer{3}, 'b');
layer{3}.dim(2) = vocabSize;


lstm_idx = ReturnLayerIdxByName(layer, 'LSTM');
shareUntil = lstm_idx + para.shareNLayersAfterLSTM;

layer2 = genNetworkFeedForward_v2(layer{shareUntil}.dim(1), [], vocabSize, 'cross_entropy');
layer2 = layer2(2:end);
layer2{1}.prev = shareUntil - length(layer) -1;
layer2{end-1}.inputIdx = 3;

layer = [layer layer2];

layer = FinishLayer(layer);
end

