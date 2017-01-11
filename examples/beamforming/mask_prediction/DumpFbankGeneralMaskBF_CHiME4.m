function DumpFbankGeneralMaskBF_CHiME4
addpath('../lib');
addpath('local');

if 0
    nCh = 6;
    modelDir = 'LSTM_Mask1chOfficial.U41972.771-1024-257.L2_3E-4.LR_3E-2';    % the directory in ./nnet to be used
    iteration = 7;      % the iteration number to be used
    nPass = 1;
    poolingType  = 'median';
    poolingType2 = 'none';
    noiseCovL2 = 0.00000;
    vadNoise = 0;
else
    nCh = 6;
%     modelDir = 'SplitMaskBF_LSTM_MVDR_DNN.U8634_mixed_randPair.771-1024-AM0-7_2048-1981.L2_0.LR_3E-2';
%     modelDir = 'MaskBF6ch_split0_LSTM_MVDR_DNN.U8634_mixed_randPair.771-1024-AM0-7_2048-1981.L2_0.LR_3E-3';    % the directory in ./nnet to be used
%     modelDir = 'MaskBF5ch_split0_LSTM_MVDR_DNN.U8634_mixed_randPair.771-1024-AM0-7_2048-1981.L2_0.LR_1E-3';
%     modelDir = 'MaskBF2ch_split0_LSTM_MVDR_DNN.U8634_mixed_randPair.771-1024-AM0-7_2048-1981.L2_0.LR_3E-2';
    modelDir = 'MaskBF2ch_split0_LSTM_MVDR_DNN.U8634_mixed_randPair.771-512-512-AM0-7_2048-1981.L2_0.LR_3E-3';
    
    iteration = 1;      % the iteration number to be used
    nPass = 1;
    poolingType  = 'median';
    poolingType2 = 'none';
    noiseCovL2 = 0.0;
    vadNoise = 0.0;
end

chime_root = ChoosePath4OS({'F:/Chime/data/', '/home/xiaoxiong/CHiME4'});   % you can set two paths, first for windows OS and second for Linux OS. 
% build the network to apply beamforming on test data. This network may be
% different from the network used during training. 
[layer, para, expTag] = BuildGeneralMaskBF_CHiME4(modelDir, iteration, nCh, poolingType, poolingType2, nPass, noiseCovL2, vadNoise);
para.local.wavroot_noisy = [chime_root '/audio/16kHz/isolated'];
mvdr = 'spacialCov';%'eigenVector';

expTag = ['LSTM_Mask_6ch_Medianpool_MVDR_' mvdr '_CE'];
para.local.fbankroot = [ './fbank/' expTag];
para.IO.inputFeature = [1 1];
% my_mkdir(para.local.fbankroot);

wavlist = LoadWavTest_CHiME4(para);
wavreader.name = 'wavfile';
wavreader.array = 1;
wavreader.multiArrayFiles = 1;
[~,nUtt] = size(wavlist);

if strcmp(mvdr, 'eigenVector')
    bf_layer_idx = ReturnLayerIdxByName(layer, 'Beamforming');
    mvdr_layer_idx = bf_layer_idx - 1;
    layer{mvdr_layer_idx}.name = 'MVDR_EigenVector';
end

for si = nUtt:-1:3281 
%for si = 3280:-1:1
    [~,uttID] = fileparts(wavlist{1,si});
    PrintProgress(si, nUtt, 100, uttID);

    [wav] = InputReader(wavlist(:,si), wavreader);
    Data(1).data{1} = StoreWavInt16(wav);
    
    output = FeatureTree2(Data, para, layer);
    
    noisy_complex_spec = output{1}{1};
    enhanced_complex_spec = output{1}(2:2+nPass-1);
    offset = nPass+1;
    weight = output{1}(offset+1:offset+nPass);
    offset = offset + nPass;
    mask = output{1}(offset+1:end-1);
    enhanced_fbank = output{1}{end};
    
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
        imagesc(CMN(log(abs([noisy_complex_spec(1:257,:); cell2mat(enhanced_complex_spec')]))')');colorbar
        
        title(regexprep(uttID, '_', '\\_'));        pause(.1);
    end
    
    words = ExtractWordsFromString_v2(dos2unix(wavlist{1,si}), '/');
    %fbank_file = [para.local.fbankroot '/' words{end-2} '/' words{end-1} '/' uttID(1:end-4) '.CH5.fbank'];     % use a fake channel 5
    fbank_file = [para.local.fbankroot '/'  words{end-1} '/' uttID(1:end-4) '.CH5.fbank'];
    dirname = fileparts(fbank_file);
    my_mkdir(dirname);
    writeHTK(fbank_file, enhanced_fbank', 'MFCC_0', 1);
end
end