
function [layer, scm_idx, split] = HandleSTFT(layer, STFT_layer_idx, nPass, para)
[scm_idx, split] = GetScmLayer(layer);
bf_idx = ReturnLayerIdxByName(layer, 'Beamforming');
for i=1:nPass
    if split
        layer{scm_idx(i)}.prev(3) = STFT_layer_idx - scm_idx(i);
    else
        layer{scm_idx(i)}.prev(2) = STFT_layer_idx - scm_idx(i);
    end
    layer{bf_idx(i)}.prev = [-1 STFT_layer_idx-bf_idx(i)];
    layer{bf_idx(i)}.freqBin = para.topology.freqBin;
end
end