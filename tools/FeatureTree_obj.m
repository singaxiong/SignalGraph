function [output, layerOut] = FeatureTree_obj(visible, para, layer)
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
visible = visible.ShuffleData(para);

output = {}; layerOut = {};
for blk_i = 1:visible.nBlock
    minibatch = visible.PrepareMinibatch(para.precision, 1, para.NET.batchSize, blk_i);
    nMinibatch = size(minibatch,2);
    
    for utt_i = 1:nMinibatch
        PrintProgress(utt_i, nMinibatch, 100);
        batch_data = GetMinibatch(minibatch, utt_i, para.useGPU);
        
        % Use mode=3 to generate network output only
        if nargout>1
            [~, layerOut{end+1}, output{end+1}] = DNN_Cost10(layer, batch_data, para,3);
        else
            [~, ~, output{end+1}] = DNN_Cost10(layer, batch_data, para,3);
        end
    end
end
end