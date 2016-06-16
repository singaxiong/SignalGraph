function SaveDNN(layer, para, LOG, modelfile)
layer = clean_network_layer(layer);
if para.useGPU
    for layer_i=1:length(layer)
        layer_fields = fields(layer{layer_i});
        for i=1:length(layer_fields)
            layer{layer_i}.(layer_fields{i}) = gather(layer{layer_i}.(layer_fields{i}));
        end
    end
    for i=1:length(para.preprocessing)
        para.preprocessing{i} = FeaturePipe2Memory(para.preprocessing{i});
    end
end
save(modelfile, 'layer', 'para', 'LOG');

end
