function [layer, para, expTag] = BuildGeneralMaskBF_CHiME4(modelDir, iteration, nCh, poolingType, poolingType2, nPass)
% modelDir =  'SplitMaskBF_LSTM_MVDR_DNN.U8634_mixed_randPair.771-1024-AM0-7_2048-1981.L2_0.LR_3E-2';
% iteration = 4;
% useChannel = 6;
% poolingType = 'Median';

expTag = [modelDir '_itr' num2str(iteration) '_' num2str(nCh) 'ch_' poolingType '_' poolingType2 '_pass' num2str(nPass)];

% switch modelDir
%     case {'SplitMaskBF_LSTM_MVDR_DNN.U8634_mixed_randPair.771-1024-AM0-7_2048-1981.L2_0.LR_3E-2',...    % it4
%             'Split0_2chMaskBF_LSTM_MVDR_DNN.U8634_mixed_randPair.771-1024-AM0-7_2048-1981.L2_0.LR_3E-3',...
%             'MaskBF_LSTM_MVDR_DNN.U8634_mixed_randPair.771-1024-AM0-7_2048-1981.L2_0.LR_3E-3',...
%             'LSTM_Mask1ch.U69953.771-1024-257.L2_3E-4.LR_3E-2'}    % it2
        modelfile = dir(['nnet/' modelDir '/nnet.itr' num2str(iteration) '.*']);
        modelfile = ['nnet/' modelDir '/' modelfile.name];
% end

dnn = load(modelfile);
para = dnn.para;
layer = dnn.layer;
CMN_idx = ReturnLayerIdxByName(layer, 'CMN');
if length(CMN_idx)==1   % this is a MSE trained mask estimator
    para.topology = SetDefaultValue(para.topology, 'nChMask', 1);
    layerCE = genNetworkMaskBF_CE(para.topology);
    scm_idx = GetScmLayer(layerCE);
    layer = [layer(1:end-2) layerCE(scm_idx:end)];
    CMN_idx = ReturnLayerIdxByName(layer, 'CMN');
    layer = layer(1:CMN_idx(end));
    layer = InitMelLayer(layer, para);
else
    layer = layer(1:CMN_idx(end));
end

para.useGPU = 0;                            % do not use GPU at test time
para.IO = RemoveIOStream(para.IO, [2]);     % as we only have one input stream at test time, strip off the second stream settings
para.IO.inputFeature(1) = 1;
para = ParseOptions2(para);
para.topology.freqBin = (0:1/para.topology.fft_len:0.5)*2*pi;
para.topology.nFreqBin = length(para.topology.freqBin);
para.topology.nCh = nCh;

% revise the first 3 layers (STFT) according to the number of channels
layer{1}.dim =  [1 1] * nCh;
layer{2}.dim =  [para.topology.nFreqBin 1] * nCh;
layer{3}.dim =  [1 1] * para.topology.nFreqBin * nCh;
layer{3}.W = layer{3}.W(1) * eye(layer{3}.dim(1));
layer{3}.b = zeros(layer{3}.dim(1),1);
STFT_layer_idx = 3;

% decide how many layers to feed into LSTM
if ~strcmpi(poolingType, 'none')    % if we use pooling, we will process each channels individually. 
    ExtractDims_idx = ReturnLayerIdxByName(layer, 'ExtractDims');
    layer{ExtractDims_idx}.name = 'reshape';    % reshape speech into a tensor of nFreqBin x nFrame x nCh
    layer{ExtractDims_idx}.sourceDims = para.topology.nFreqBin;
    layer{ExtractDims_idx}.targetDims = [para.topology.nFreqBin nCh];
    layer{ExtractDims_idx}.dim = [1 nCh] * para.topology.nFreqBin;
    permuteLayer.name = 'permute';
    permuteLayer.permute_order = [1 3 2];
    permuteLayer.prev = -1;
    permuteLayer.dim = [1 1]*layer{ExtractDims_idx}.dim(1);    
    layer = [layer(1:ExtractDims_idx) permuteLayer layer(ExtractDims_idx+1:end)];
end

% add pooling layers
if ~strcmpi(poolingType, 'none')
    lstm_idx =  ReturnLayerIdxByName(layer, 'LSTM');
    pool_layer.name = lower(poolingType);
    pool_layer.prev = -1;
    pool_layer.pool_idx = 3;     % pool over the third dimension, i.e. overal all channels
    pool_layer.dim = [1 1] * para.topology.nFreqBin;
    
    [scm_layer, split] = GetScmLayer(layer); 
    if ~split
        layer2 = [layer(1:lstm_idx+2) pool_layer layer(lstm_idx+3:end)];
        scm_layer = ReturnLayerIdxByName(layer, 'SpatialCovMask');
        layer2{scm_layer}.prev = [-1 STFT_layer_idx-scm_layer];
    else
        layer2 = [layer(1:lstm_idx+2) pool_layer layer(lstm_idx+3:lstm_idx+4) pool_layer layer(lstm_idx+5:end)];
        scm_layer = ReturnLayerIdxByName(layer2, 'SpatialCovSplitMask');
        layer2{scm_layer}.prev = [-4 -1 STFT_layer_idx-scm_layer];
        pool_idx = ReturnLayerIdxByName(layer2, poolingType);
        layer2{pool_idx(1)+1}.prev = -4;
        layer2{pool_idx(2)+1}.prev = [-4 -1 STFT_layer_idx-pool_idx(2)-1];
    end
    [layer2, scm_idx, split] = HandleSTFT(layer2, STFT_layer_idx, 1, para);
else
    layer2 = layer;
end

bf_idx = ReturnLayerIdxByName(layer2, 'beamforming');
layer2{bf_idx}.freqBin = para.topology.freqBin;
layer2{bf_idx}.prev = [-1 STFT_layer_idx-bf_idx];

% add multiple passes of mask estimation
if nPass>1
    power_idx = ReturnLayerIdxByName(layer, 'power');
    layerBF_withoutPooling = layer(power_idx(1)+1:power_idx(2));
    power_idx = ReturnLayerIdxByName(layer2, 'power');
    layerBF = layer2(1:power_idx(2));
    layerFE = layer2(power_idx(2)+1:end);
    
    layerCascaded = layerBF;
    for i=1:nPass-1
        layerCascaded = [layerCascaded layerBF_withoutPooling];
    end
    layerCascaded = [layerCascaded layerFE];

    [layerCascaded, scm_idx, split] = HandleSTFT(layerCascaded, STFT_layer_idx, nPass, para);
    
    if strcmpi(poolingType2, 'mean')   % pool between the first pass mask and second pass mask
        % find the first and second pass masks
        add_layer.name = 'add';
        add_layer.prev = [];
        add_layer.dim = [1 1] * para.topology.nFreqBin;

        for pass = 2:nPass
            [scm_idx, split] = GetScmLayer(layerCascaded);
            prev_pass_mask_idx = layerCascaded{scm_idx(1)}.prev(1:2)+scm_idx(1);
            curr_pass_mask_idx = layerCascaded{scm_idx(pass)}.prev(1:2)+scm_idx(pass);
            
            break_idx = curr_pass_mask_idx(2);
            layerCascaded = [layerCascaded(1:break_idx) add_layer add_layer layerCascaded(break_idx+1:end)];
            add_idx = break_idx+[1 2];
            for i=1:2
                layerCascaded{add_idx(i)}.prev = [prev_pass_mask_idx(i) curr_pass_mask_idx(i)] - add_idx(i);
            end
            layerCascaded{break_idx+3}.prev(1) = -2;
        end
        
        [layerCascaded, scm_idx, split] = HandleSTFT(layerCascaded, STFT_layer_idx, nPass, para);
    end
    layer = layerCascaded;
else
    layer = layer2;
end

para.out_layer_idx = [STFT_layer_idx ReturnLayerIdxByName(layer, 'beamforming')];
[scm_idx, split] = GetScmLayer(layer); 
mask_idx = [];
for i=1:length(scm_idx)
    if split
        mask_idx = [mask_idx layer{scm_idx(i)}.prev(1:2)+scm_idx(i)];
    else
        mask_idx = [mask_idx layer{scm_idx(i)}.prev(1)+scm_idx(i)];
    end
end
para.out_layer_idx = [para.out_layer_idx ReturnLayerIdxByName(layer, 'MVDR_spatialCov') mask_idx length(layer)];
layer = FinishLayer(layer);
end

    