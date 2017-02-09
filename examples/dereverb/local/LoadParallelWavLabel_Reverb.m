% Load far talk, close talk, and frame label. 
function [Data, para, vocab] = LoadParallelWavLabel_Reverb(para, step, dataset, datatype, distance)
nCh = para.topology.useChannel;
wavlist = []; wavlistClean = [];
% note that training data do not contain real recordings. 
% real dev and eval data do not have clean version
for type_i = 1:length(datatype)
    for dist_i = 1:length(distance)
        for roomID = 1:3
            [tasklist, taskfile] = LoadTaskFile_Reverb([para.local.wavroot '/taskFiles'], dataset, datatype{type_i}, distance{dist_i}, nCh, roomID);
            wavlist = [wavlist Tasklist2wavlist(para.local.wavroot, tasklist, dataset, datatype{type_i})];
            
            if strcmpi(dataset, 'train')
                wavlistClean = findFiles([para.local.wsjcam0root '/data_wav/primary_microphone/si_tr'], para.local.wsjcam0ext);
                break;
            end
            [tasklistClean, taskfileClean] = LoadTaskFile_Reverb([para.local.wavroot '/taskFiles'], dataset, datatype{type_i}, 'cln', nCh, roomID);
            wavlistClean = [wavlistClean Tasklist2wavlist(para.local.wavroot, tasklistClean, dataset, datatype{type_i})];
        end
        if strcmpi(dataset, 'train'); break; end    % For training data, far and near distances are in one list, so we can stop here.
    end
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
    if strcmpi(dataset, 'train')
        wavfileClean = clean_struct.(['U_' clean_uttID]);
        wav_c = audioread(wavfileClean)';
    else
        [wav_c] = InputReader(wavlistClean(:,si), wavreader);
    end
    wav_c = StoreWavInt16(wav_c);
    
    % synchronize the length of label and wav
    nFr_feat = enframe_decide_frame_number(size(wav,2), frame_size, frame_shift);
    nFr_feat = min(nFr_feat, enframe_decide_frame_number(size(wav_c,2), frame_size, frame_shift));
    nFr_label = 10000;
    nFr = min([nFr_feat nFr_label]);
    requiredLen = DecideWavLen4XFrames(nFr, frame_size, frame_shift);
    wav(:,requiredLen+1:end) = [];
    wav_c(:,requiredLen+1:end) = [];
    
    if para.local.useFileName
        for i=1:nCh
            wavfileArray{i} = sprintf('%s 0 %2.3f', wavlist{i,si}, size(wav,2)/fs);
        end
        wav_noisy{end+1} = wavfileArray;
        if strcmpi(dataset, 'train')
            wav_clean{end+1} = sprintf('%s 0 %2.3f', clean_struct.(['U_' clean_uttID]), size(wav,2)/fs);
        else
            wav_clean{end+1} = sprintf('%s 0 %2.3f', wavlistClean{1,si}, size(wav,2)/fs);
        end
    else
        wav_noisy{end+1} = wav;
        wav_clean{end+1} = wav_c;
    end
    if para.local.loadLabel
        label{end+1} = [];
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
    para.IO.fileReader(1) = wavreader;
    para.IO.fileReader(2) = wavreader;
    para.IO.fileReader(2).array = 0;
    para.IO.fileReader(2).multiArrayFiles = 0;
end
if para.local.loadLabel
    Data(3).data = label;
    para.IO.inputFeature(3) = 1;
    para.IO.frame_rate(3) = 100;
    para.IO.isTensor(3) = 1;
    para.IO.fileReader(3).name = '';
end
end
