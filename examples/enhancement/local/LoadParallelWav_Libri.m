% Load far talk, close talk, and frame label. 
function [Data, para, vocab] = LoadParallelWav_Libri(para, step)
wavlist = findFiles([para.local.cv_wav_root '/distorted'], 'wav');
wavlist = wavlist(:,step:step:end);

wavreader.name = 'wavfile';
wavreader.array = 0;
wavreader.precision = 'int16';
frame_size = 400;
frame_shift = 160;
fs=para.topology.fs;

% Load data file list
wav_noisy = {}; wav_clean = {};
nUtt = size(wavlist,2);
for si = 1:nUtt
    words = ExtractWordsFromString_v2(wavlist{1,si}, '/');
    curr_uttID = words{end}(1:end-4);
    PrintProgress(si, nUtt, 100, curr_uttID);

    [wav] = InputReader(wavlist{si}, wavreader);
    cleanFile = [para.local.cv_wav_root '/clean/' curr_uttID '.wav'];
    [wav_c] = InputReader(cleanFile, wavreader);
    
    % synchronize the length of label and wav
    nFr_feat = enframe_decide_frame_number(size(wav,2), frame_size, frame_shift);
    nFr_feat = min(nFr_feat, enframe_decide_frame_number(size(wav_c,2), frame_size, frame_shift));
    nFr_label = 10000;
    nFr = min([nFr_feat nFr_label]);
    requiredLen = DecideWavLen4XFrames(nFr, frame_size, frame_shift);
    wav(:,requiredLen+1:end) = [];
    wav_c(:,requiredLen+1:end) = [];
    
    if para.local.useFileName
        wav_noisy{end+1} = sprintf('%s 0 %2.3f', wavlist{si}, size(wav,2)/fs);
        wav_clean{end+1} = sprintf('%s 0 %2.3f', cleanFile, size(wav,2)/fs);
    else    
        wav_noisy{end+1} = wav;
        wav_clean{end+1} = wav_c;
    end

end

Data(1).data = wav_noisy;
Data(2).data = wav_clean;

para.IO.inputFeature = [1 1];
para.IO.DataSyncSet{1} = [];
para.IO.frame_rate = [16000 16000];
para.IO.isTensor = [1 1];
if para.local.useFileName
    para.IO.inputFeature([1 2]) = 0;
    wavreader.precision = 'int16';
    if isfield(para.IO, 'fileReader')
        para.IO = rmfield(para.IO, 'fileReader');
    end
    para.IO.fileReader(1) = wavreader;
    para.IO.fileReader(2) = wavreader;
    para.IO.fileReader(2).array = 0;
    para.IO.fileReader(2).multiArrayFiles = 0;
end
end
