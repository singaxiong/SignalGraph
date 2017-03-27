
function [model] = PrepareNetwork4Separation(dnnFile, hasClean, useGPU)
dnn = load(dnnFile);

layer = dnn.layer;
para = dnn.para;
para.local.useFileName = 1;
% para.topology.useFileName = 1;
% para.local.seglen = 100;
% para.local.segshift = 100;
para.useGPU = useGPU;

% mixture STFT
stft_idx = ReturnLayerIdxByName(layer, 'stft');
para.out_layer_idx(1) = stft_idx(1);

% mixture log spec
log_idx = ReturnLayerIdxByName(layer, 'log');
para.out_layer_idx(2) = log_idx(1);

% enhanced log spec
hadamard_idx = ReturnLayerIdxByName(layer, 'hadamard');
if ~isempty(hadamard_idx)
    for i=1:2
        para.out_layer_idx(end+1) = hadamard_idx(i)+layer{hadamard_idx(i)}.next;
    end
    for i=1:2
        para.out_layer_idx(end+1) = hadamard_idx(i)+layer{hadamard_idx(i)}.prev(1);
    end
    para.test.mask_idx = [-1 0] + length(para.out_layer_idx);
else
    cmn_idx = ReturnLayerIdxByName(layer, 'cmn');
    if ~isempty(cmn_idx)
        for i=2:3
            para.out_layer_idx = [para.out_layer_idx cmn_idx(i)+layer{cmn_idx(i)}.prev];
        end
    else
        delta_idx = ReturnLayerIdxByName(layer, 'delta');
        para.out_layer_idx = [para.out_layer_idx delta_idx(2:3)-1];
    end
    para.test.mask_idx = [];
end

% clean STFT and log spec
input_idx = ReturnLayerIdxByName(layer, 'input');
if hasClean==0
    layer = layer(1:input_idx(end)-1);
    para.IO = RemoveIOStream(para.IO, 2);
    para.test.clean_idx = [];
    para.test.clean_stft_idx = [];
else
    para.out_layer_idx = [para.out_layer_idx log_idx(end-1:end)];
    para.test.clean_idx = [-1 0] + length(para.out_layer_idx);
    para.out_layer_idx = [para.out_layer_idx stft_idx(end-1:end)];
    para.test.clean_stft_idx = [-1 0] + length(para.out_layer_idx);   
end

model.layer= layer;
model.para = para;
end