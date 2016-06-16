function output = FeatureTree2(visible, para, layer)
% remove target and cost function nodes if any
for i=1:length(layer)
    if strcmpi(layer{i}.name, 'target') || strcmpi(layer{i}.name, 'cross_entropy') || strcmpi(layer{i}.name, 'mse') || strcmpi(layer{i}.name, 'logistic')
        layer{i}.name = 'ignore';
    end
    if i > max(para.out_layer_idx)  % we don't compute layers that is after the last output layer
        layer{i}.name = 'ignore';
    end
end
para.NET.sentenceMinibatch = 1;
[minibatch] = MinibatchPackaging_tree4(visible, para);

output = {};
for utt_i = 1:minibatch.nBatch
    PrintProgress(utt_i, minibatch.nBatch, 100);
    batch_data = GetMinibatch2(minibatch, para, utt_i);
    
    % Use mode=3 to generate network output only
    [~, ~, output{utt_i}] = DNN_Cost10(layer, batch_data, para,3);
end
end
