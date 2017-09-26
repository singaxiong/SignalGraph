% function RunMVDR
addpath('../lib', 'local', '../mask_prediction/local');
para.topology.nCh = 6;
para = ConfigBasicSTFT(para);

chime_root = ChoosePath4OS({'E:/Data/CHiME4', '/home/xiaoxiong/CHiME4'});   % you can set two paths, first for windows OS and second for Linux OS.
para.local.wavroot_noisy = [chime_root '/audio/isolated'];

wavlist = LoadWavTest_CHiME4(para);
wavreader.name = 'wavfile';
wavreader.array = 1;
wavreader.multiArrayFiles = 1;
[~,nUtt] = size(wavlist);

% Load the LSTM-mask based MVDR
nCh = 6;
if 0
    modelDir = '../mask_prediction/nnet/LSTM_Mask1chOfficial.U41972.771-1024-257.L2_3E-4.LR_3E-2';    % the directory in ./nnet to be used
    iteration = 7;      % the iteration number to be used
elseif 0
    modelDir = '../mask_prediction/nnet/SplitMaskBF_LSTM_MVDR_DNN.U8634_mixed_randPair.771-1024-AM0-7_2048-1981.L2_0.LR_3E-2';    % the directory in ./nnet to be used
    iteration = 5;      % the iteration number to be used
else
    modelDir = '../mask_prediction/nnet/MaskBF5ch_split0_MTL3E-3_LSTM_MVDR_DNN.U8634_mixed_randPair.771-1024-AM0-7_2048-1981.L2_0.LR_1E-2';
    iteration = 4;
end
nPass = 1;
poolingType  = 'median';
poolingType2 = 'none';
noiseCovL2 = 0.00000;
vadNoise = 0;
[layerMask, paraMask, expTagMask] = BuildGeneralMaskBF_CHiME4(modelDir, iteration, nCh, poolingType, poolingType2, nPass, noiseCovL2, vadNoise);


layer = genNetworkMVDR(para.topology);
para.out_layer_idx = length(layer);
para.IO.nStream = 1;
para.NET.sequential = 1;
para = ParseOptions2(para);

for si = nUtt:-1:3281
    [~,uttID] = fileparts(wavlist{1,si});
    PrintProgress(si, nUtt, 100, uttID);
    
    [wav] = InputReader(wavlist(:,si), wavreader);
    Data(1).data{1} = StoreWavInt16(wav);
    
    output = FeatureTree2(Data, paraMask, layerMask);
    
    noisy_complex_spec = output{1}{1};
    enhanced_complex_spec = output{1}(2:2+nPass-1);
    offset = nPass+1;
    weight = output{1}(offset+1:offset+nPass);
    offset = offset + nPass;
    mask = output{1}(offset+1:end-1);
    enhanced_fbank = output{1}{end};

    layer{3}.W = mask{1};
    layer{4}.W = 1-mask{1};
    output2 = FeatureTree2(Data, para, layer);
    enhanced_complex_spec2 = output2{1}{1};
        
    if 0   % generate waveform
        wav_noisy = complexSpec2wav(noisy_complex_spec(1:nBin,:)', 400, 240);
        wav_dnn_bf = complexSpec2wav(enhanced_complex_spec', 400, 240);
        sound(wav_noisy/max(wav_noisy)*2,fs);
        sound(wav_dnn_bf/max(wav_dnn_bf)*2,fs);
    end
    
    if ispc && mod(si,10)==-1   % check enhanced spectrogram
        figure(1);
        if length(mask)==nPass
            for i=1:length(mask)
                subplot(nPass,1,i); imagesc(mask{i});
            end
        else
            for i=1:length(mask)
                subplot(nPass,2,i); imagesc(mask{i});
            end
        end
        figure(2)
        imagesc(CMN(log(abs([noisy_complex_spec(1:257,:); cell2mat(enhanced_complex_spec'); enhanced_complex_spec2]))')');colorbar
        
        title(regexprep(uttID, '_', '\\_'));        pause(.1);
    end
    
end





