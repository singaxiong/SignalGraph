% Load far talk, close talk, and frame label. 
function [Data, para, vocab] = LoadParallelWavLabel_CHiME4(para, step, dataset)
switch lower(dataset)
    case {'dt05','et05'}
        wavlist = ['../Kaldi/data/' dataset '_multi_noisy/wav.scp'];
        ali_file = [para.local.aliDir '_dt05/ali.txt'];
    case 'tr05'
        wavlist = ['../Kaldi/data/' dataset '_multi_noisy/wav.scp'];
        ali_file = [para.local.aliDir '/ali.txt'];
        wavlist_clean = findFiles(para.local.wavroot_clean, 'wav');
        for i=1:length(wavlist_clean)   % build an index of clean files so we can find them quickly by utterance ID
            [~,curr_uttID] = fileparts(wavlist_clean{i});
            words = ExtractWordsFromString_v2(curr_uttID, '_');
            clean_struct.(['U_' words{2}]) = wavlist_clean{i};
        end
end

[ali, vocab] = LoadKaldiFrameLabel(ali_file);
wavlist = my_cat(wavlist);
wavlist = wavlist(step:step:end);

wavreader.name = 'wavfile';
wavreader.array = 1;
wavreader.multiArrayFiles = 1;

nCh = 6;
fs = 16000;   
frame_size = fs*0.025;
frame_shift = fs*0.01;

% Load data file list
wav_noisy = {};  label = {};  wav_clean = {};
for si = 1:length(wavlist)
    words = ExtractWordsFromString_v2(wavlist{si});
    curr_uttID = words{1};
    wavfile = words{2};
    PrintProgress(si, length(wavlist), 100, curr_uttID);
    
    % get alignment
    if ~strcmpi(para.local.data, 'mixed') && isempty(regexp(lower(curr_uttID), para.local.data))
        continue;
    end
    
    words = ExtractWordsFromString_v2(curr_uttID, '_');
    curr_uttID2 = [words{2} '_' words{3}(1:3) '_' words{4}];
    clean_uttID = words{2};
    type = words{end};
    fieldname = ['U_' curr_uttID2];
    if ~isfield(ali, fieldname); continue; end    % if there is no label for current utterance, skip it.
    curr_label = ali.(fieldname);
    curr_label = curr_label(:)'+1;  % convert to row vector and the index starts with 1
    
    words = ExtractWordsFromString_v2(wavfile, '/');
    wavfileRoot = [para.local.wavroot_noisy '/' words{end-1} '/' words{end}(1:end-5)];
    
    if strcmpi(type, 'simu')
        wavfile_clean = clean_struct.(['U_' clean_uttID]);
    elseif strcmpi(type, 'real')
        wavfile_clean = [wavfileRoot '0.wav'];
    end
    
    % read in the waveform
    % for the two channel track, we may choose a random pair, or use all
    % the channel pairs. 
    if length(para.topology.useChannel)==1
        if para.topology.useChannel==6
            ch_idx = randperm(6);
        elseif para.topology.useChannel==5
            ch_idx = randperm(6);
            ch_idx(ch_idx==2) = [];     % we don't use channel 2
        elseif strcmpi(para.local.pair, 'randPair')
            ch_idx = randperm(nCh);
            ch_idx(ch_idx==2) = [];     % we don't use channel 2
            ch_idx = sort(ch_idx(1:para.topology.useChannel));
        elseif strcmpi(para.local.pair, 'allPair')
            ch_idx = [1 3; 1 4; 1 5; 1 6; 3 4; 3 5; 3 6; 4 5; 4 6; 5 6];
        end
    else
        ch_idx = para.topology.useChannel;
    end
    
    for pi = 1:size(ch_idx,1)   % add all channel pairs to the training data
        clear wavfileArray
        for i=1:size(ch_idx,2)
            wavfileArray{i} = [wavfileRoot num2str(ch_idx(pi,i)) '.wav'];
        end
        [wav] = InputReader(wavfileArray, wavreader);
        wav = StoreWavInt16(wav);
        wav_c = audioread(wavfile_clean);
        wav_c = StoreWavInt16(wav_c)';
        
        % synchronize the length of label and wav
        nFr_feat = enframe_decide_frame_number(size(wav,2), frame_size, frame_shift);
        nFr_feat = min(nFr_feat, enframe_decide_frame_number(size(wav_c,2), frame_size, frame_shift));
        nFr_label = length(curr_label);
        nFr = min([nFr_feat nFr_label]);
        requiredLen = DecideWavLen4XFrames(nFr, frame_size, frame_shift);
        wav(:,requiredLen+1:end) = [];
        wav_c(:,requiredLen+1:end) = [];
        curr_label = curr_label(1:nFr);

        if para.local.useFileName
            for i=1:size(ch_idx,2)
                wavfileArray{i} = sprintf('%s 0 %2.3f', wavfileArray{i}, size(wav,2)/fs);
            end
            wav_noisy{end+1} = wavfileArray;
            wavfileClean = sprintf('%s 0 %2.3f', wavfile_clean, size(wav,2)/fs);
            wav_clean{end+1} = wavfileClean;
        else
            wav_noisy{end+1} = wav;
            wav_clean{end+1} = wav_c;
        end
        label{end+1} = curr_label;
    end
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
