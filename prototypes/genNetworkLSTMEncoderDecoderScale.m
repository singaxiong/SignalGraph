function layer = genNetworkLSTMEncoderDecoderScale(para)

layer = genNetworkLSTMEncoderDecoder(para);

layerScale.name = 'Input';
layerScale.inputIdx = 3;
layerScale.dim = [1 1];
layerScale.next = 1;

layer = [layer(1:end-1) layerScale layer(end)];

layer{end-3}.next = 3;
layer{end-2}.next = 2;
layer{end}.prev = [-3 -2 -1];
% layer{end} = rmfield(layer{end}, 'costFrameSelection');

end

