function tgt_model = CopyNetWeightsByLayerIdx(src_model, src_layer_idx, tgt_model, tgt_layer_idx)

for i=1:length(src_layer_idx)
    
    idx1 = src_layer_idx(i);
    idx2 = tgt_layer_idx(i);
    
    if isfield(src_model{idx1}, 'W') && isfield(tgt_model{idx2}, 'W')
        tgt_model{idx2}.W = src_model{idx1}.W;
    end

    if isfield(src_model{idx1}, 'b') && isfield(tgt_model{idx2}, 'b')
        tgt_model{idx2}.b = src_model{idx1}.b;
    end
end
