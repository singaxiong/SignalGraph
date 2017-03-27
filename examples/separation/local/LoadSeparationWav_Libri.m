% Load far talk, close talk, and frame label. 
function [Data, para, vocab] = LoadSeparationWav_Libri(para, step)
if isfield(para.local, 'cv_wav_root_clean')
    wavlistClean = findFiles(para.local.cv_wav_root_clean, para.local.cv_wav_clean_ext);
    wavlist = findFiles([para.local.cv_wav_root], 'wav');
    wavIndexClean = fileIndexByUttID(wavlistClean);
else    
    wavlist = findFiles([para.local.cv_wav_root '/mixed'], 'wav');
end
wavlist = wavlist(:,step:step:end);

wavreader.name = 'wavfile';
wavreader.array = 0;
wavreader.precision = 'int16';
frame_size = 400;
frame_shift = 160;
fs=para.topology.fs;
seglen = para.local.seglen;
segshift = para.local.segshift;

% Load data file list
wav_noisy = {};  wav_clean1 = {}; wav_clean2 = {};
nUtt = size(wavlist,2);
for si = 1:nUtt
    words = ExtractWordsFromString_v2(wavlist{1,si}, '/');
    curr_uttID = words{end}(1:end-4);
    PrintProgress(si, nUtt, 100, curr_uttID);

    [wav] = InputReader(wavlist{si}, wavreader);
    if isfield(para.local, 'cv_wav_root_clean')
        cleanFile = searchUttIDInFileIndex(wavIndexClean, curr_uttID);
    else
        cleanFile1 = regexprep(wavlist{1,si}, 'mixed', 'clean1');
        cleanFile2 = regexprep(wavlist{1,si}, 'mixed', 'clean2');
    end
    [wav_c1] = InputReader(cleanFile1, wavreader);
    [wav_c2] = InputReader(cleanFile2, wavreader);
    
    % synchronize the length of label and wav
    nFr_feat = enframe_decide_frame_number(size(wav,2), frame_size, frame_shift);
    nFr_feat = min(nFr_feat, enframe_decide_frame_number(size(wav_c1,2), frame_size, frame_shift));
    nFr_feat = min(nFr_feat, enframe_decide_frame_number(size(wav_c2,2), frame_size, frame_shift));
    nFr_label = 10000;
    nFr = min([nFr_feat nFr_label]);
    requiredLen = DecideWavLen4XFrames(nFr, frame_size, frame_shift);
    wav(:,requiredLen+1:end) = [];
    wav_c1(:,requiredLen+1:end) = [];
    wav_c2(:,requiredLen+1:end) = [];
    
    if para.local.useFileName
        wav_noisy{end+1} = sprintf('%s 0 %2.3f', wavlist{si}, size(wav,2)/fs);
        wav_clean1{end+1} = sprintf('%s 0 %2.3f', cleanFile1, size(wav,2)/fs);
        wav_clean2{end+1} = sprintf('%s 0 %2.3f', cleanFile2, size(wav,2)/fs);
    else    
        if 1
            wav_c_seg1 = DivideSent2Segments(wav_c1, (seglen-1)*frame_shift+frame_size, segshift*frame_shift, 1);
            wav_c_seg2 = DivideSent2Segments(wav_c2, (seglen-1)*frame_shift+frame_size, segshift*frame_shift, 1);
            wav_seg = DivideSent2Segments(wav, (seglen-1)*frame_shift+frame_size, segshift*frame_shift, 1);
            wav_noisy = [wav_noisy; wav_seg];
            wav_clean1 = [wav_clean1; wav_c_seg1];
            wav_clean2 = [wav_clean2; wav_c_seg2];
        else
            wav_noisy{end+1} = wav;
            wav_clean1{end+1} = wav_c1;
            wav_clean2{end+1} = wav_c2;
        end
    end

end

Data(1).data = wav_noisy;
Data(2).data = wav_clean1;
Data(3).data = wav_clean2;

para.IO.inputFeature = [1 1 1];
para.IO.DataSyncSet{1} = [];
para.IO.frame_rate = [16000 16000 16000];
para.IO.isTensor = [1 1 1];
if para.local.useFileName
    para.IO.inputFeature([1 2 3]) = 0;
    wavreader.precision = 'int16';
    if isfield(para.IO, 'fileReader')
        para.IO = rmfield(para.IO, 'fileReader');
    end
    para.IO.fileReader(1) = wavreader;
    para.IO.fileReader(2) = wavreader;
    para.IO.fileReader(3) = wavreader;
    para.IO.fileReader(2).array = 0;
    para.IO.fileReader(2).multiArrayFiles = 0;
    para.IO.fileReader(3).array = 0;
    para.IO.fileReader(3).multiArrayFiles = 0;
end
end
