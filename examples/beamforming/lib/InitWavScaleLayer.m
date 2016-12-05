function layer = InitWavScaleLayer(layer, scale, para)
fft_layer_idx = ReturnLayerIdxByName(layer, 'stft');
for i=fft_layer_idx
    dim = layer{i}.dim(1);
    layer{i+1}.W = eye(dim)*scale;
    layer{i+1}.b = zeros(dim,1);
end

end