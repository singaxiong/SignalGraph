
function [model] = PrepareNetwork4Enhancement(dnnFile, hasClean, useMasking, useGPU)
dnn = load(dnnFile);

layer = dnn.layer;
para = dnn.para;
para.local.useFileName = 1;
% para.topology.useFileName = 1;
% para.local.seglen = 100;
% para.local.segshift = 100;
para.useGPU = useGPU;

% noisy STFT
stft_idx = ReturnLayerIdxByName(layer, 'stft');
para.out_layer_idx(1) = stft_idx(1);

% noisy log spec
log_idx = ReturnLayerIdxByName(layer, 'log');
para.out_layer_idx(2) = log_idx(1);

% enhanced log spec
if useMasking
    hadamard_idx = ReturnLayerIdxByName(layer, 'hadamard');
    para.out_layer_idx(end+1) = hadamard_idx;
    para.out_layer_idx(end+1) = layer{hadamard_idx}.prev(1) + hadamard_idx;
    para.test.mask_idx = length(para.out_layer_idx);
else
    para.out_layer_idx(end+1) = length(layer)+layer{end}.prev(1);
    para.test.mask_idx = [];
end

% variance
if strcmpi(layer{end}.name, 'll_gaussian')
    para.out_layer_idx(end+1) = length(layer)+layer{end}.prev(2);
    para.test.var_idx = length(para.out_layer_idx);
else
    para.test.var_idx = [];
end

% clean STFT and log spec
input_idx = ReturnLayerIdxByName(layer, 'input');
if hasClean==0
    layer = layer(1:input_idx(end)-1);
    para.IO = RemoveIOStream(para.IO, 2);
    para.test.clean_idx = [];
    para.test.clean_stft_idx = [];
else
    para.out_layer_idx(end+1) = log_idx(end);
    para.test.clean_idx = length(para.out_layer_idx);
    para.out_layer_idx(end+1) = stft_idx(end); 
    para.test.clean_stft_idx = length(para.out_layer_idx);   
end

model.layer= layer;
model.para = para;
end