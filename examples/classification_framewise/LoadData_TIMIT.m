function [Data_tr, Data_cv, para] = LoadData_TIMIT(para, dataset)

wavroot = ['G:/Data/TIMIT/SPEECHDATA/' upper(dataset)];
vocab_file = 'TIMIT_vocab_phone.mat';
vocab = LoadTIMITVocab(vocab_file, wavroot, 'phone');
if para.local.V == 61
    para.local.vocab = vocab;
elseif para.local.V == 48
    para.local.vocab = unique(TIMIT_map61to48(vocab));
elseif para.local.V == 39
    para.local.vocab = unique(TIMIT_map48to39(unique(TIMIT_map61to48(vocab))));
end

cv_step = 40;

fbank_tr = {}; fbank_cv = {};
label_tr = {}; label_cv = {};

wavlist = findFiles(wavroot, 'wav');
for i=1:length(wavlist)
    PrintProgress(i, length(wavlist), 100);
    [~,uttID] = fileparts(wavlist{i});
    if strcmpi(uttID(1:2), 'SA'); continue; end     % do not use SA sentences as they are the same across all speakers.
    [wav,fs] = audioread(wavlist{i});
    
    if para.local.fs == 8000
        fbank = wav2fbank(wav(2:2:end), 8000, 0.01, 40)';
    else
        fbank = wav2fbank(wav, fs, 0.01, 40)';
    end
    
    nFr = size(fbank,2);

    if para.local.doCMN
        fbank = CMN(fbank')';
    end
    
    phone_label_txt = my_cat([wavlist{i}(1:end-3) 'PHN']);
    clear phone_label
    for j=1:length(phone_label_txt)
        words = ExtractWordsFromString_v2(phone_label_txt{j});
        phone_label.start(j) = max(1,round(str2num(words{1})/fs*100));
        phone_label.stop(j) = max(1,round(str2num(words{2})/fs*100));
        phone_label.label{j} = words{3};
    end
    if phone_label.start(1)>1   % sometimes, the starting time starts from a positive number rather than 0
        phone_label.start = [1 phone_label.start];
        phone_label.stop = [phone_label.start(2)-1 phone_label.stop];
        phone_label.label = ['h#' phone_label.label];   % assume the initial unlabeled part is silence
    end
    
    if para.local.V == 61
        phone_label.labelMapped = phone_label.label;
    elseif para.local.V == 48
        phone_label.labelMapped = TIMIT_map61to48(phone_label.label);
    elseif para.local.V == 39
        phone_label.labelMapped = TIMIT_map48to39(TIMIT_map61to48(phone_label.label));
    end
    
    phone_label.label = label2idx(para.local.vocab, phone_label.labelMapped);
    phone_label_num = seg2label(phone_label);
    
    if strcmpi(dataset, 'train') && mod(i,cv_step)==0
        fbank_cv{end+1} = fbank;
        label_cv{end+1} = phone_label_num;
    else
        fbank_tr{end+1} = fbank;
        label_tr{end+1} = phone_label_num;
    end
end

Data_tr(1).data = fbank_tr;
Data_cv(1).data = fbank_cv;
Data_tr(2).data = label_tr;
Data_cv(2).data = label_cv;

para.IO.context = [9 1];
para.IO.DataSyncSet{1} = [1 2];     % note that the feature and label may not have the same length, we need to synchronize them
para.IO.frame_rate = [100 100];     % this is the number of samples per second (frame rate) of the streams. This information will be used in synchronization
end
