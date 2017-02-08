% Load far talk, close talk, and frame label. 
function [Data, para, vocab] = LoadParallelWavLabel_Reverb(para, step, dataset, datatype, distance)
nCh = para.topology.useChannel;
wavlist = []; wavlistClean = [];
for roomID = 1:3
    [tasklist, taskfile] = LoadTaskFile_Reverb([para.local.wavroot '/taskFiles'], dataset, datatype, distance, nCh, roomID);
    wavlist = [wavlist Tasklist2wavlist(para.local.wavroot, tasklist, dataset, datatype)];

    [tasklistClean, taskfileClean] = LoadTaskFile_Reverb([para.local.wavroot '/taskFiles'], dataset, datatype, 'cln', nCh, roomID);
    wavlistClean = [wavlistClean Tasklist2wavlist(para.local.wavroot, tasklistClean, dataset, datatype)];
end
for i=1:length(wavlistClean)   % build an index of clean files so we can find them quickly by utterance ID
    [~,curr_uttID] = fileparts(wavlistClean{i});
    clean_struct.(['U_' curr_uttID]) = wavlistClean{i};
end

% Currently we don't support loading of frame label yet. 
% switch lower(dataset)
%     case {'eval_simu'}
%         ali_file = [para.local.aliDir '_' dataset '/ali.txt'];
%     case 'tr05'
%         ali_file = [para.local.aliDir '/ali.txt'];
% end
% [ali, vocab] = LoadKaldiFrameLabel(ali_file);
vocab = [];

wavlist = wavlist(:,step:step:end);
wavlistClean = wavlistClean(:,step:step:end);

wavreader.name = 'wavfile';
wavreader.array = 1;
wavreader.multiArrayFiles = 1;

fs = 16000;   
frame_size = fs*0.025;
frame_shift = fs*0.01;

% Load data file list
wav_noisy = {};  label = {};  wav_clean = {};
nUtt = size(wavlist,2);
for si = 1:nUtt
    words = ExtractWordsFromString_v2(wavlist{1,si}, '/');
    curr_uttID = words{end}(1:end-4);
    PrintProgress(si, nUtt, 100, curr_uttID);
    
    words2 = ExtractWordsFromString_v2(curr_uttID, '_');
    clean_uttID = words2{1};

    [wav] = InputReader(wavlist(:,si), wavreader);
    wav = StoreWavInt16(wav);
    [wav_c] = InputReader(wavlistClean(:,si), wavreader);
    wav_c = StoreWavInt16(wav_c);
    
    % synchronize the length of label and wav
    nFr_feat = enframe_decide_frame_number(size(wav,2), frame_size, frame_shift);
    nFr_feat = min(nFr_feat, enframe_decide_frame_number(size(wav_c,2), frame_size, frame_shift));
    nFr_label = 10000;
%     nFr_label = length(curr_label);
    nFr = min([nFr_feat nFr_label]);
    requiredLen = DecideWavLen4XFrames(nFr, frame_size, frame_shift);
    wav(:,requiredLen+1:end) = [];
    wav_c(:,requiredLen+1:end) = [];
    curr_label = zeros(1,nFr);
%     curr_label = curr_label(1:nFr);
    
    if para.local.useFileName
        for i=1:nCh
            wavfileArray{i} = sprintf('%s 0 %2.3f', wavlist{i,si}, size(wav,2)/fs);
        end
        wav_noisy{end+1} = wavfileArray;
        wav_clean{end+1} = sprintf('%s 0 %2.3f', wavlistClean{1,si}, size(wav,2)/fs);
    else
        wav_noisy{end+1} = wav;
        wav_clean{end+1} = wav_c;
    end
    label{end+1} = curr_label;
end

Data(1).data = wav_noisy;
Data(2).data = label;
Data(3).data = wav_clean;

para.IO.inputFeature = [1 1 1];
para.IO.DataSyncSet{1} = [];
para.IO.frame_rate = [16000 100 16000];
para.IO.isTensor = [1 1 1];
if para.local.useFileName
    para.IO.inputFeature([1 3]) = 0;
    wavreader.precision = 'int16';
    para.IO.fileReader(1) = wavreader;
    para.IO.fileReader(2).name = '';
    para.IO.fileReader(3) = wavreader;
    para.IO.fileReader(3).array = 0;
    para.IO.fileReader(3).multiArrayFiles = 0;
end
end
