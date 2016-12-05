function [Data, para] = LoadWavMask_Simu_CHiME4(para, step, dataset)
nCh = 6;
chime_root = 'F:/Data2/CHiME4';
% ChoosePath4OS allows us to define two paths for the data, one for
% Windows system and one for Linux system. The function will select the
% correct path, so we don't need to change code for different platforms.
wavroot_noisy = ChoosePath4OS({[chime_root '/audio/isolated'], '/home/xiaoxiong/CHiME4/isolated'});
wavlist_noisy = unique(my_cat('local/wavlist_tr05_simu'));
wavroot_clean = ChoosePath4OS({[chime_root '/audio/isolated/tr05_org'], '/home/xiaoxiong/CHiME4/isolated/tr05_org'});
wavlist_clean = findFiles(wavroot_clean, 'wav');

for i=1:length(wavlist_clean)   % build an index of clean files so we can find them quickly by utterance ID
    [~,curr_uttID] = fileparts(wavlist_clean{i});
    words = ExtractWordsFromString_v2(curr_uttID, '_');
    clean_struct.(['U_' words{2}]) = wavlist_clean{i};
end

wavlist_noisy = wavlist_noisy(1:step:end);  % choose a portion of the data
if strcmpi(dataset,'train')     % split to training and cross validation data
    wavlist_noisy(50:50:end) = [];
else
    wavlist_noisy = wavlist_noisy(50:50:end);
end

% Load data file list
for i=1:length(wavlist_noisy)
    PrintProgress(i, length(wavlist_noisy), 500);
    % provide the file name of the noisy speech,     
    if para.topology.nChMask>1
        ch_idx = randperm(nCh);
        ch_idx(ch_idx==2) = [];     % we don't use channel 2
        ch_idx = sort(ch_idx(1:para.topology.nChMask));
        clear wav wavfile
        for j=1:length(ch_idx)
            wavfile{j} = [wavroot_noisy '/' wavlist_noisy{i} 'CH' num2str(ch_idx(j)) '.wav'];
            if j==1 || ~para.topology.useFileName
                [wav(:,j),fs] = audioread(wavfile{j});
            end
        end
        wav_noisy = wav(:,1);
    else
        wavfile{1} = [wavroot_noisy '/' wavlist_noisy{i}];
        [wav, fs] = audioread(wavfile{1});
        wav_noisy = wav;
    end

    [~,curr_uttID] = fileparts(wavlist_noisy{i});
    words = ExtractWordsFromString_v2(curr_uttID, '_');
    clean_uttID = words{2};
    wavfile_clean = clean_struct.(['U_' clean_uttID]);
    wav_clean = audioread(wavfile_clean);
    
    % Compute the mask based on local SNR estimate
    
    % The clean signal and noisy signal have different gains, so we need to
    % first roughly normalize the absolute power of clean speech in wav_noisy and
    % wav_clean
    noise_power0 = mean(wav_noisy(1:fs*0.2).^2);
    noisy_power0 = mean(wav_noisy.^2);
    clean_power0 = noisy_power0 - noise_power0;
    clean_power_true = mean(wav_clean.^2);
    scale = sqrt(max(0,clean_power0) / clean_power_true);
    scale = max(scale, 1);
    wav_clean = wav_clean*scale;
    
    [~,spec_clean] = wav2abs(wav_clean,fs);
    [~,spec_noisy] = wav2abs(wav_noisy,fs);
    nFr = min(size(spec_clean,2), size(spec_noisy,2));
    spec_clean = spec_clean(1:257,1:nFr);
    spec_noisy = spec_noisy(1:257,1:nFr);
    spec_noise = spec_noisy-spec_clean;
    power_clean = abs(spec_clean).^2;
    power_noisy = abs(spec_noisy).^2;
    if 0
        power_noise = power_noisy - power_clean;       % use the noisy-clean as noise estimate. This is not stable as sometimes the noisy and clean are not in the same scale
    else
        power_noise = repmat(mean(abs(spec_noisy(:,1:20)).^2,2), 1, nFr);   % use the first 20 frames as noise estimate
    end
    SNR = 10*log10(power_clean ./ power_noise);
    mask{i} = logical(SNR>5);   % 0dB threshold gives us too many speech TF bins. So 5dB is used as threshold. The threshold may not be critical, as we are only using the mask to initialize the mask subnet. 
    if mod(i,100)==0
        subplot(5,1,1:2); imagesc(log([abs(spec_clean).^2; abs(spec_noisy).^2; power_noise]));
        subplot(5,1,3); imagesc(SNR);
        subplot(5,1,4); imagesc(mask{i});
        subplot(5,1,5); plot(wav_noisy); hold on; plot(wav_clean,'r'); hold off;
        pause(.01);
    end
    
    % make sure that the features generated from the wave have the same
    % length as the mask
    nSampleRequired = DecideWavLen4XFrames(nFr, para.topology.frame_len, para.topology.frame_shift);
    if para.topology.useFileName
        for j=1:length(wavfile)
            wavfile{j} = sprintf('%s 0 %2.3f', wavfile{j}, nSampleRequired/fs);
        end
        wavInt{i} = wavfile;
    else
        wavInt{i} = StoreWavInt16(wav(1:nSampleRequired,:))';      % note that the input waveform to the network should be a row vector
    end
end

Data(1).data = wavInt;
Data(2).data = mask;
para.IO.context = [1 1];
para.IO.sparse = [0 0];
para.IO.DataSyncSet{1} = [];
para.IO.frame_rate = [100 100];
para.IO.isTensor = [1 1];

if para.topology.useFileName
    para.IO.inputFeature = [0 1];
    para.IO.fileReader(1).name = 'wavfile';
    para.IO.fileReader(1).multiArrayFiles = 1;
    para.IO.fileReader(1).array = 1;
    para.IO.fileReader(1).fs = 16000;
    para.IO.fileReader(1).precision = 'int16';
    para.IO.fileReader(2).name = '';
else
    para.IO.inputFeature = [1 1];
end
end
