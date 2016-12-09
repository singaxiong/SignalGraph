function DumpFbankGeneralMaskBF_CHiME4
addpath('../lib');
addpath('local');

nCh = 6;
modelDir = 'F:\Dropbox\Workspace\Enhancement\Beamforming\CHiME4\XX\nnet\LSTM_Mask1chOfficial.U41972.771-1024-257.L2_3E-4.LR_3E-2';    % the directory in ./nnet to be used
iteration = 8;      % the iteration number to be used
nPass = 1;
poolingType  = 'median';
poolingType2 = 'none';

[layer, para, expTag] = BuildGeneralMaskBF_CHiME4(modelDir, iteration, nCh, poolingType, poolingType2, nPass);

wavlist = LoadCHiME4_test(nCh);
wavreader.name = 'wavfile';
wavreader.array = 1;
wavreader.multiArrayFiles = 1;

fbankroot = ChoosePath4OS({['F:/Data/CHiME3/fbank/' expTag], ['/home/xiaoxiong/CHiME3/fbank/' expTag]});
my_mkdir(fbankroot);

[nCh,nUtt] = size(wavlist);
for si = 1:nUtt
    [~,uttID] = fileparts(wavlist{1,si});
    PrintProgress(si, nUtt, 100, uttID);

    [wav] = InputReader(wavlist(:,si), wavreader);
    wav = StoreWavInt16(wav);
    if nCh==6
        wav = wav([1 2 3 4 5 6], :);
    end

    if para.topology.useWav
        Data(1).data{1} = wav;
    end
    
    output = FeatureTree2(Data, para, layer);
    
    nBin = para.topology.fft_len/2+1;
    nCh = para.topology.nCh;

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
%         imagesc(enhanced_fbank);colorbar
        
        title(regexprep(uttID, '_', '\\_'));        pause(.1);
    end
    
    words = ExtractWordsFromString_v2(dos2unix(wavlist{1,si}), '/');
    fbank_file = [fbankroot '/' words{end-2} '/' words{end-1} '/' uttID(1:end-4) '.CH5.fbank'];     % use a fake channel 5
    dirname = fileparts(fbank_file);
    my_mkdir(dirname);
    writeHTK(fbank_file, enhanced_fbank', 'MFCC_0', 1);
    
end
end
